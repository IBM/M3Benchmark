from fastapi import APIRouter, HTTPException, Query, Request
import sqlite3

app = APIRouter()

conn = sqlite3.connect('db/image_and_language/image_and_language.sqlite')
cursor = conn.cursor()

# Endpoint to get the count of object samples for a given image ID
@app.get("/v1/image_and_language/count_object_samples_by_image_id", operation_id="get_count_object_samples", summary="Retrieves the total number of object samples associated with a specific image, identified by its unique image ID.")
async def get_count_object_samples(img_id: int = Query(..., description="ID of the image")):
    cursor.execute("SELECT COUNT(OBJ_SAMPLE_ID) FROM IMG_OBJ WHERE IMG_ID = ?", (img_id,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of images with more than a specified number of object samples
@app.get("/v1/image_and_language/count_images_with_object_samples_greater_than", operation_id="get_count_images_with_object_samples", summary="Retrieves the total count of images that have more than the specified minimum number of object samples. This operation is useful for understanding the distribution of object samples across images.")
async def get_count_images_with_object_samples(min_object_samples: int = Query(..., description="Minimum number of object samples")):
    cursor.execute("SELECT COUNT(T1.IMG_ID) FROM ( SELECT IMG_ID FROM IMG_OBJ GROUP BY IMG_ID HAVING COUNT(OBJ_SAMPLE_ID) > ? ) T1", (min_object_samples,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the image ID with the highest number of object samples
@app.get("/v1/image_and_language/image_with_most_object_samples", operation_id="get_image_with_most_object_samples", summary="Retrieves the image identifier that has the most object samples. This operation returns the image ID with the highest count of associated object samples, providing a quick way to identify the most populated image in terms of object samples.")
async def get_image_with_most_object_samples():
    cursor.execute("SELECT IMG_ID FROM IMG_OBJ GROUP BY IMG_ID ORDER BY COUNT(OBJ_SAMPLE_ID) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"img_id": []}
    return {"img_id": result[0]}

# Endpoint to get object sample IDs for a given image ID and object class ID
@app.get("/v1/image_and_language/object_sample_ids_by_image_and_class", operation_id="get_object_sample_ids", summary="Retrieves the unique identifiers of object samples associated with a specific image and object class. The operation requires the image ID and the object class ID as input parameters to filter the results.")
async def get_object_sample_ids(img_id: int = Query(..., description="ID of the image"), obj_class_id: int = Query(..., description="ID of the object class")):
    cursor.execute("SELECT OBJ_SAMPLE_ID FROM IMG_OBJ WHERE IMG_ID = ? AND OBJ_CLASS_ID = ?", (img_id, obj_class_id))
    result = cursor.fetchall()
    if not result:
        return {"object_sample_ids": []}
    return {"object_sample_ids": [row[0] for row in result]}

# Endpoint to get the count of image relations where object sample IDs match for a given image ID
@app.get("/v1/image_and_language/count_image_relations_with_matching_object_samples", operation_id="get_count_image_relations", summary="Retrieves the total count of image relations for a specific image where the object sample IDs are identical. This operation is useful for understanding the prevalence of matching object samples within a given image.")
async def get_count_image_relations(img_id: int = Query(..., description="ID of the image")):
    cursor.execute("SELECT SUM(CASE WHEN IMG_ID = ? THEN 1 ELSE 0 END) FROM IMG_REL WHERE OBJ1_SAMPLE_ID = OBJ2_SAMPLE_ID", (img_id,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the coordinates of objects in an image where object sample IDs match
@app.get("/v1/image_and_language/object_coordinates_with_matching_object_samples", operation_id="get_object_coordinates", summary="Retrieve the coordinates of objects in a specific image where the object sample IDs are identical. The image is identified by its unique ID.")
async def get_object_coordinates(img_id: int = Query(..., description="ID of the image")):
    cursor.execute("SELECT T2.X, T2.Y, T2.W, T2.H FROM IMG_REL AS T1 INNER JOIN IMG_OBJ AS T2 ON T1.IMG_ID = T2.IMG_ID WHERE T1.IMG_ID = ? AND T1.OBJ1_SAMPLE_ID = T1.OBJ2_SAMPLE_ID", (img_id,))
    result = cursor.fetchall()
    if not result:
        return {"coordinates": []}
    return {"coordinates": [{"x": row[0], "y": row[1], "w": row[2], "h": row[3]} for row in result]}

# Endpoint to get the count of a specific object class in an image
@app.get("/v1/image_and_language/count_object_class_in_image", operation_id="get_count_object_class", summary="Retrieves the total count of a specified object class within a particular image. The operation requires the object class name and the image ID as input parameters. It processes the data from the OBJ_CLASSES and IMG_OBJ tables, filtering by the provided object class and image ID, and returns the sum of instances of the specified object class in the image.")
async def get_count_object_class(obj_class: str = Query(..., description="Object class name"), img_id: int = Query(..., description="ID of the image")):
    cursor.execute("SELECT SUM(CASE WHEN T1.OBJ_CLASS = ? THEN 1 ELSE 0 END) FROM OBJ_CLASSES AS T1 INNER JOIN IMG_OBJ AS T2 ON T1.OBJ_CLASS_ID = T2.OBJ_CLASS_ID WHERE T2.IMG_ID = ?", (obj_class, img_id))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of images containing a specific object class
@app.get("/v1/image_and_language/count_images_with_object_class", operation_id="get_count_images_with_object_class", summary="Retrieves the total number of images that contain a specified object class. The object class is identified by its name, which is provided as an input parameter. The operation does not return individual image details, but rather a single count value.")
async def get_count_images_with_object_class(obj_class: str = Query(..., description="Object class name")):
    cursor.execute("SELECT COUNT(T.IMG_ID) FROM ( SELECT T2.IMG_ID FROM OBJ_CLASSES AS T1 INNER JOIN IMG_OBJ AS T2 ON T1.OBJ_CLASS_ID = T2.OBJ_CLASS_ID WHERE T1.OBJ_CLASS = ? GROUP BY T2.IMG_ID ) T", (obj_class,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the object classes in a given image
@app.get("/v1/image_and_language/object_classes_in_image", operation_id="get_object_classes", summary="Retrieve the distinct object classes associated with a specific image. The operation identifies the object classes by matching the provided image ID with the corresponding entries in the image-object mapping table. The result is a list of unique object classes present in the image.")
async def get_object_classes(img_id: int = Query(..., description="ID of the image")):
    cursor.execute("SELECT T1.OBJ_CLASS FROM OBJ_CLASSES AS T1 INNER JOIN IMG_OBJ AS T2 ON T1.OBJ_CLASS_ID = T2.OBJ_CLASS_ID WHERE T2.IMG_ID = ? GROUP BY T1.OBJ_CLASS", (img_id,))
    result = cursor.fetchall()
    if not result:
        return {"object_classes": []}
    return {"object_classes": [row[0] for row in result]}

# Endpoint to get the predicate classes for a given image and object sample IDs
@app.get("/v1/image_and_language/predicate_classes_by_image_and_object_samples", operation_id="get_predicate_classes", summary="Retrieves the predicate classes associated with a specific image and two object samples. The image and object samples are identified by their respective IDs. This operation returns the predicate classes that connect the image and the two object samples based on their relationships in the database.")
async def get_predicate_classes(img_id: int = Query(..., description="ID of the image"), obj1_sample_id: int = Query(..., description="ID of the first object sample"), obj2_sample_id: int = Query(..., description="ID of the second object sample")):
    cursor.execute("SELECT T1.PRED_CLASS FROM PRED_CLASSES AS T1 INNER JOIN IMG_REL AS T2 ON T1.PRED_CLASS_ID = T2.PRED_CLASS_ID WHERE T2.IMG_ID = ? AND T2.OBJ1_SAMPLE_ID = ? AND T2.OBJ2_SAMPLE_ID = ?", (img_id, obj1_sample_id, obj2_sample_id))
    result = cursor.fetchall()
    if not result:
        return {"predicate_classes": []}
    return {"predicate_classes": [row[0] for row in result]}

# Endpoint to get the sum of a specific prediction class for a given image ID where object sample IDs are different
@app.get("/v1/image_and_language/sum_pred_class_img_id_diff_obj_ids", operation_id="get_sum_pred_class_img_id_diff_obj_ids", summary="Retrieves the total count of instances of a specified prediction class for a given image, considering only those instances where the object sample IDs are different.")
async def get_sum_pred_class_img_id_diff_obj_ids(pred_class: str = Query(..., description="Prediction class"), img_id: int = Query(..., description="Image ID")):
    cursor.execute("SELECT SUM(CASE WHEN T1.PRED_CLASS = ? THEN 1 ELSE 0 END) FROM PRED_CLASSES AS T1 INNER JOIN IMG_REL AS T2 ON T1.PRED_CLASS_ID = T2.PRED_CLASS_ID WHERE T2.IMG_ID = ? AND T2.OBJ1_SAMPLE_ID != T2.OBJ2_SAMPLE_ID", (pred_class, img_id))
    result = cursor.fetchone()
    if not result:
        return {"sum": []}
    return {"sum": result[0]}

# Endpoint to get the prediction class for given object sample IDs
@app.get("/v1/image_and_language/pred_class_obj_sample_ids", operation_id="get_pred_class_obj_sample_ids", summary="Retrieve the predicted class for a pair of object sample IDs. This operation fetches the prediction class from the database by joining the PRED_CLASSES and IMG_REL tables using the provided object sample IDs. The result is the predicted class associated with the given object sample ID pair.")
async def get_pred_class_obj_sample_ids(obj1_sample_id: int = Query(..., description="Object 1 sample ID"), obj2_sample_id: int = Query(..., description="Object 2 sample ID")):
    cursor.execute("SELECT T1.PRED_CLASS FROM PRED_CLASSES AS T1 INNER JOIN IMG_REL AS T2 ON T1.PRED_CLASS_ID = T2.PRED_CLASS_ID WHERE T2.OBJ1_SAMPLE_ID = ? AND T2.OBJ2_SAMPLE_ID = ?", (obj1_sample_id, obj2_sample_id))
    result = cursor.fetchall()
    if not result:
        return {"pred_classes": []}
    return {"pred_classes": [row[0] for row in result]}

# Endpoint to get the sum of a specific prediction class where object sample IDs are different
@app.get("/v1/image_and_language/sum_pred_class_diff_obj_ids", operation_id="get_sum_pred_class_diff_obj_ids", summary="Retrieves the total count of instances for a specified prediction class where the object sample IDs are not identical. This operation is useful for analyzing prediction class distribution across different object samples.")
async def get_sum_pred_class_diff_obj_ids(pred_class: str = Query(..., description="Prediction class")):
    cursor.execute("SELECT SUM(CASE WHEN T1.PRED_CLASS = ? THEN 1 ELSE 0 END) FROM PRED_CLASSES AS T1 INNER JOIN IMG_REL AS T2 ON T1.PRED_CLASS_ID = T2.PRED_CLASS_ID WHERE T2.OBJ1_SAMPLE_ID != T2.OBJ2_SAMPLE_ID", (pred_class,))
    result = cursor.fetchone()
    if not result:
        return {"sum": []}
    return {"sum": result[0]}

# Endpoint to get image IDs for a specific prediction class where object sample IDs are different and the count of image IDs is greater than a specified number
@app.get("/v1/image_and_language/img_ids_pred_class_diff_obj_ids_count", operation_id="get_img_ids_pred_class_diff_obj_ids_count", summary="Retrieves image IDs for a specified prediction class where the object sample IDs are distinct and the count of image IDs surpasses a given threshold.")
async def get_img_ids_pred_class_diff_obj_ids_count(pred_class: str = Query(..., description="Prediction class"), count: int = Query(..., description="Count of image IDs")):
    cursor.execute("SELECT T2.IMG_ID FROM PRED_CLASSES AS T1 INNER JOIN IMG_REL AS T2 ON T1.PRED_CLASS_ID = T2.PRED_CLASS_ID WHERE T1.PRED_CLASS = ? AND T2.OBJ1_SAMPLE_ID != T2.OBJ2_SAMPLE_ID GROUP BY T2.IMG_ID HAVING COUNT(T2.IMG_ID) > ?", (pred_class, count))
    result = cursor.fetchall()
    if not result:
        return {"img_ids": []}
    return {"img_ids": [row[0] for row in result]}

# Endpoint to get the prediction class for a given image ID where object sample IDs are the same
@app.get("/v1/image_and_language/pred_class_img_id_same_obj_ids", operation_id="get_pred_class_img_id_same_obj_ids", summary="Retrieve the predicted class for an image, given that the object sample IDs in the image are identical. The operation requires the image ID as input.")
async def get_pred_class_img_id_same_obj_ids(img_id: int = Query(..., description="Image ID")):
    cursor.execute("SELECT T1.PRED_CLASS FROM PRED_CLASSES AS T1 INNER JOIN IMG_REL AS T2 ON T1.PRED_CLASS_ID = T2.PRED_CLASS_ID WHERE T2.IMG_ID = ? AND T2.OBJ1_SAMPLE_ID = T2.OBJ2_SAMPLE_ID", (img_id,))
    result = cursor.fetchall()
    if not result:
        return {"pred_classes": []}
    return {"pred_classes": [row[0] for row in result]}

# Endpoint to get the coordinates and dimensions of objects for a given image ID and prediction class
@app.get("/v1/image_and_language/obj_coords_dimensions_img_id_pred_class", operation_id="get_obj_coords_dimensions_img_id_pred_class", summary="Retrieves the coordinates and dimensions of objects in an image, based on a specified image ID and prediction class. The operation filters objects by their association with the provided image ID and prediction class, and returns their X, Y, width, and height values.")
async def get_obj_coords_dimensions_img_id_pred_class(img_id: int = Query(..., description="Image ID"), pred_class: str = Query(..., description="Prediction class")):
    cursor.execute("SELECT T3.X, T3.Y, T3.W, T3.H FROM PRED_CLASSES AS T1 INNER JOIN IMG_REL AS T2 ON T1.PRED_CLASS_ID = T2.PRED_CLASS_ID INNER JOIN IMG_OBJ AS T3 ON T2.OBJ1_SAMPLE_ID = T3.OBJ_CLASS_ID WHERE T2.IMG_ID = ? AND T1.PRED_CLASS = ?", (img_id, pred_class))
    result = cursor.fetchall()
    if not result:
        return {"coords_dimensions": []}
    return {"coords_dimensions": [{"x": row[0], "y": row[1], "w": row[2], "h": row[3]} for row in result]}

# Endpoint to get the average Y coordinate for a specific prediction class and image ID where object sample IDs are different
@app.get("/v1/image_and_language/avg_y_coord_pred_class_img_id_diff_obj_ids", operation_id="get_avg_y_coord_pred_class_img_id_diff_obj_ids", summary="Retrieves the average Y coordinate for a specified prediction class and image, considering only instances where the object sample IDs are different. This operation is useful for analyzing the vertical distribution of objects within an image, based on a particular prediction class.")
async def get_avg_y_coord_pred_class_img_id_diff_obj_ids(pred_class: str = Query(..., description="Prediction class"), img_id: int = Query(..., description="Image ID")):
    cursor.execute("SELECT CAST(SUM(T3.Y) AS REAL) / COUNT(CASE WHEN T1.PRED_CLASS = ? THEN 1 ELSE NULL END) FROM PRED_CLASSES AS T1 INNER JOIN IMG_REL AS T2 ON T1.PRED_CLASS_ID = T2.PRED_CLASS_ID INNER JOIN IMG_OBJ AS T3 ON T2.OBJ1_SAMPLE_ID = T3.OBJ_CLASS_ID WHERE T2.IMG_ID = ? AND T2.OBJ1_SAMPLE_ID != T2.OBJ2_SAMPLE_ID", (pred_class, img_id))
    result = cursor.fetchone()
    if not result:
        return {"avg_y_coord": []}
    return {"avg_y_coord": result[0]}

# Endpoint to get the percentage of a specific object class for a given image ID
@app.get("/v1/image_and_language/percentage_obj_class_img_id", operation_id="get_percentage_obj_class_img_id", summary="Retrieves the percentage of a specified object class present in a given image. This operation calculates the proportion of the specified object class relative to all object classes identified in the image. The input parameters include the object class and the image ID.")
async def get_percentage_obj_class_img_id(obj_class: str = Query(..., description="Object class"), img_id: int = Query(..., description="Image ID")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T1.OBJ_CLASS = ? THEN 1 ELSE NULL END) AS REAL) * 100 / COUNT(T1.OBJ_CLASS_ID) FROM OBJ_CLASSES AS T1 INNER JOIN IMG_OBJ AS T2 ON T1.OBJ_CLASS_ID = T2.OBJ_CLASS_ID WHERE T2.IMG_ID = ?", (obj_class, img_id))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the count of attribute classes
@app.get("/v1/image_and_language/count_att_classes", operation_id="get_count_att_classes", summary="Retrieves the total count of distinct attribute classes available in the system. This operation does not require any input parameters and returns a single integer value representing the count.")
async def get_count_att_classes():
    cursor.execute("SELECT COUNT(ATT_CLASS_ID) FROM ATT_CLASSES")
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of object classes
@app.get("/v1/image_and_language/count_obj_classes", operation_id="get_count_obj_classes", summary="Retrieves the total count of distinct object classes available in the system. This operation does not require any input parameters and returns a single integer value representing the count.")
async def get_count_obj_classes():
    cursor.execute("SELECT COUNT(OBJ_CLASS_ID) FROM OBJ_CLASSES")
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of prediction classes
@app.get("/v1/image_and_language/pred_class_count", operation_id="get_pred_class_count", summary="Retrieves the total count of unique prediction classes available in the system. This operation does not require any input parameters and returns a single integer value representing the count.")
async def get_pred_class_count():
    cursor.execute("SELECT COUNT(PRED_CLASS_ID) FROM PRED_CLASSES")
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get object dimensions based on image ID and object class
@app.get("/v1/image_and_language/object_dimensions", operation_id="get_object_dimensions", summary="Retrieves the dimensions (x, y, width, height) of specific objects within an image. The operation requires the unique identifier of the image (img_id) and the class of the object (obj_class) to filter the results accordingly.")
async def get_object_dimensions(img_id: int = Query(..., description="Image ID"), obj_class: str = Query(..., description="Object class")):
    cursor.execute("SELECT T2.X, T2.Y, T2.W, T2.H FROM OBJ_CLASSES AS T1 INNER JOIN IMG_OBJ AS T2 ON T1.OBJ_CLASS_ID = T2.OBJ_CLASS_ID WHERE T2.IMG_ID = ? AND T1.OBJ_CLASS = ?", (img_id, obj_class))
    result = cursor.fetchall()
    if not result:
        return {"dimensions": []}
    return {"dimensions": result}

# Endpoint to get the count of a specific attribute class in an image
@app.get("/v1/image_and_language/attribute_class_count", operation_id="get_attribute_class_count", summary="Retrieve the total count of a specified attribute class within a given image. This operation calculates the sum of instances where the attribute class matches the provided class, for the specified image.")
async def get_attribute_class_count(img_id: int = Query(..., description="Image ID"), att_class: str = Query(..., description="Attribute class")):
    cursor.execute("SELECT SUM(CASE WHEN T2.ATT_CLASS = ? THEN 1 ELSE 0 END) FROM IMG_OBJ_ATT AS T1 INNER JOIN ATT_CLASSES AS T2 ON T1.ATT_CLASS_ID = T2.ATT_CLASS_ID WHERE T1.IMG_ID = ?", (att_class, img_id))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get object sample IDs based on image ID, prediction class, and another object sample ID
@app.get("/v1/image_and_language/obj_sample_id_by_image_pred_class_and_obj", operation_id="get_obj_sample_id_by_image_pred_class_and_obj", summary="Retrieves the IDs of object samples that share the same image ID, prediction class, and object 2 sample ID. This operation is useful for identifying related object samples based on their common attributes.")
async def get_obj_sample_id_by_image_pred_class_and_obj(img_id: int = Query(..., description="Image ID"), pred_class: str = Query(..., description="Prediction class"), obj2_sample_id: int = Query(..., description="Object 2 sample ID")):
    cursor.execute("SELECT T2.OBJ1_SAMPLE_ID FROM PRED_CLASSES AS T1 INNER JOIN IMG_REL AS T2 ON T1.PRED_CLASS_ID = T2.PRED_CLASS_ID WHERE T2.IMG_ID = ? AND T1.PRED_CLASS = ? AND T2.OBJ2_SAMPLE_ID = ?", (img_id, pred_class, obj2_sample_id))
    result = cursor.fetchall()
    if not result:
        return {"obj_sample_ids": []}
    return {"obj_sample_ids": [row[0] for row in result]}

# Endpoint to get the count of object samples based on image ID and object class
@app.get("/v1/image_and_language/obj_sample_count_by_image_and_class", operation_id="get_obj_sample_count_by_image_and_class", summary="Retrieves the total count of object samples associated with a specific image and object class. The image is identified by its unique ID, and the object class is specified by its name. This operation provides a quantitative measure of the occurrence of a particular object class within a given image.")
async def get_obj_sample_count_by_image_and_class(img_id: int = Query(..., description="Image ID"), obj_class: str = Query(..., description="Object class")):
    cursor.execute("SELECT COUNT(T2.OBJ_SAMPLE_ID) FROM OBJ_CLASSES AS T1 INNER JOIN IMG_OBJ AS T2 ON T1.OBJ_CLASS_ID = T2.OBJ_CLASS_ID WHERE T2.IMG_ID = ? AND T1.OBJ_CLASS = ?", (img_id, obj_class))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get prediction classes based on object classes and image ID
@app.get("/v1/image_and_language/pred_class_by_object_classes_and_image", operation_id="get_pred_class_by_object_classes_and_image", summary="Retrieves the predicted classes for an image based on the provided object classes. The operation filters the image's objects by the given object classes and returns the corresponding predicted classes. The image is identified by its unique ID.")
async def get_pred_class_by_object_classes_and_image(obj_class1: str = Query(..., description="First object class"), obj_class2: str = Query(..., description="Second object class"), img_id: int = Query(..., description="Image ID")):
    cursor.execute("SELECT T1.PRED_CLASS FROM PRED_CLASSES AS T1 INNER JOIN IMG_REL AS T2 ON T1.PRED_CLASS_ID = T2.PRED_CLASS_ID INNER JOIN IMG_OBJ AS T3 ON T2.OBJ1_SAMPLE_ID = T3.OBJ_SAMPLE_ID INNER JOIN OBJ_CLASSES AS T4 ON T3.OBJ_CLASS_ID = T4.OBJ_CLASS_ID WHERE (T4.OBJ_CLASS = ? OR T4.OBJ_CLASS = ?) AND T2.IMG_ID = ? GROUP BY T1.PRED_CLASS", (obj_class1, obj_class2, img_id))
    result = cursor.fetchall()
    if not result:
        return {"pred_classes": []}
    return {"pred_classes": [row[0] for row in result]}

# Endpoint to get attribute classes based on object class and image ID
@app.get("/v1/image_and_language/att_class_by_object_class_and_image", operation_id="get_att_class_by_object_class_and_image", summary="Retrieves the attribute classes associated with a specific object class and image. The operation filters the attribute classes based on the provided object class and image ID, returning only those that match the given criteria.")
async def get_att_class_by_object_class_and_image(obj_class: str = Query(..., description="Object class"), img_id: int = Query(..., description="Image ID")):
    cursor.execute("SELECT T2.ATT_CLASS FROM IMG_OBJ_att AS T1 INNER JOIN ATT_CLASSES AS T2 ON T1.ATT_CLASS_ID = T2.ATT_CLASS_ID INNER JOIN IMG_OBJ AS T3 ON T1.IMG_ID = T3.IMG_ID INNER JOIN OBJ_CLASSES AS T4 ON T3.OBJ_CLASS_ID = T4.OBJ_CLASS_ID WHERE T4.OBJ_CLASS = ? AND T1.IMG_ID = ?", (obj_class, img_id))
    result = cursor.fetchall()
    if not result:
        return {"att_classes": []}
    return {"att_classes": [row[0] for row in result]}

# Endpoint to get object classes based on attribute class and image ID
@app.get("/v1/image_and_language/obj_class_by_att_class_and_image", operation_id="get_obj_class_by_att_class_and_image", summary="Retrieves the object classes associated with a given attribute class and image ID. The operation identifies the attribute class and image, then returns the corresponding object classes.")
async def get_obj_class_by_att_class_and_image(att_class: str = Query(..., description="Attribute class"), img_id: int = Query(..., description="Image ID")):
    cursor.execute("SELECT T4.OBJ_CLASS_ID, T4.OBJ_CLASS FROM IMG_OBJ_att AS T1 INNER JOIN ATT_CLASSES AS T2 ON T1.ATT_CLASS_ID = T2.ATT_CLASS_ID INNER JOIN IMG_OBJ AS T3 ON T1.IMG_ID = T3.IMG_ID INNER JOIN OBJ_CLASSES AS T4 ON T3.OBJ_CLASS_ID = T4.OBJ_CLASS_ID WHERE T2.ATT_CLASS = ? AND T1.IMG_ID = ?", (att_class, img_id))
    result = cursor.fetchall()
    if not result:
        return {"obj_classes": []}
    return {"obj_classes": [{"obj_class_id": row[0], "obj_class": row[1]} for row in result]}

# Endpoint to get the object class based on image ID and object sample ID
@app.get("/v1/image_and_language/get_object_class", operation_id="get_object_class", summary="Retrieves the class of an object associated with a specific image and object sample. The operation uses the provided image ID and object sample ID to identify the object and return its class.")
async def get_object_class(img_id: int = Query(..., description="Image ID"), obj_sample_id: int = Query(..., description="Object sample ID")):
    cursor.execute("SELECT T1.OBJ_CLASS FROM OBJ_CLASSES AS T1 INNER JOIN IMG_OBJ AS T2 ON T1.OBJ_CLASS_ID = T2.OBJ_CLASS_ID WHERE T2.IMG_ID = ? AND T2.OBJ_SAMPLE_ID = ?", (img_id, obj_sample_id))
    result = cursor.fetchone()
    if not result:
        return {"object_class": []}
    return {"object_class": result[0]}

# Endpoint to get the ratio of object sample IDs for two different object classes
@app.get("/v1/image_and_language/get_object_class_ratio", operation_id="get_object_class_ratio", summary="Retrieves the ratio of sample IDs for two specified object classes. This operation compares the number of sample IDs associated with the first object class to the number of sample IDs associated with the second object class. The result is a ratio that provides insights into the distribution of sample IDs across the two object classes.")
async def get_object_class_ratio(obj_class1: str = Query(..., description="First object class"), obj_class2: str = Query(..., description="Second object class")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T1.OBJ_CLASS = ? THEN T2.OBJ_SAMPLE_ID ELSE NULL END) AS REAL) / COUNT(CASE WHEN T1.OBJ_CLASS = ? THEN T2.OBJ_SAMPLE_ID ELSE NULL END) times FROM OBJ_CLASSES AS T1 INNER JOIN IMG_OBJ AS T2 ON T1.OBJ_CLASS_ID = T2.OBJ_CLASS_ID", (obj_class1, obj_class2))
    result = cursor.fetchone()
    if not result:
        return {"ratio": []}
    return {"ratio": result[0]}

# Endpoint to get the count of images with more than a specified number of attribute classes
@app.get("/v1/image_and_language/count_images_with_attribute_classes", operation_id="get_count_images_with_attribute_classes", summary="Retrieves the total count of images that have more than a specified minimum number of attribute classes. This operation is useful for understanding the distribution of images based on the number of associated attribute classes.")
async def get_count_images_with_attribute_classes(min_att_classes: int = Query(..., description="Minimum number of attribute classes")):
    cursor.execute("SELECT COUNT(*) FROM ( SELECT IMG_ID FROM IMG_OBJ_att GROUP BY IMG_ID HAVING COUNT(ATT_CLASS_ID) > ? ) T1", (min_att_classes,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct image IDs where object sample IDs are the same
@app.get("/v1/image_and_language/distinct_image_ids_same_object_sample", operation_id="get_distinct_image_ids_same_object_sample", summary="Retrieves a list of unique image identifiers (IMG_ID) where the object sample identifiers (OBJ1_SAMPLE_ID and OBJ2_SAMPLE_ID) match. This operation is useful for identifying images that share the same object sample, providing insights into image-object relationships.")
async def get_distinct_image_ids_same_object_sample():
    cursor.execute("SELECT DISTINCT IMG_ID FROM IMG_REL WHERE OBJ1_SAMPLE_ID = OBJ2_SAMPLE_ID")
    result = cursor.fetchall()
    if not result:
        return {"image_ids": []}
    return {"image_ids": [row[0] for row in result]}

# Endpoint to get the image ID with the highest number of attribute classes
@app.get("/v1/image_and_language/image_with_most_attribute_classes", operation_id="get_image_with_most_attribute_classes", summary="Retrieves the image identifier associated with the highest count of distinct attribute classes. This operation returns the image that has been annotated with the most diverse set of attributes.")
async def get_image_with_most_attribute_classes():
    cursor.execute("SELECT COUNT(ATT_CLASS_ID) FROM IMG_OBJ_att GROUP BY IMG_ID ORDER BY COUNT(ATT_CLASS_ID) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"image_id": []}
    return {"image_id": result[0]}

# Endpoint to get object class IDs for a list of object classes
@app.get("/v1/image_and_language/object_class_ids", operation_id="get_object_class_ids", summary="Retrieves the unique identifiers for a specified set of object classes. The operation accepts up to five object classes as input and returns their corresponding identifiers from the database.")
async def get_object_class_ids(obj_class1: str = Query(..., description="First object class"), obj_class2: str = Query(..., description="Second object class"), obj_class3: str = Query(..., description="Third object class"), obj_class4: str = Query(..., description="Fourth object class"), obj_class5: str = Query(..., description="Fifth object class")):
    cursor.execute("SELECT OBJ_CLASS_ID FROM OBJ_CLASSES WHERE OBJ_CLASS IN (?, ?, ?, ?, ?)", (obj_class1, obj_class2, obj_class3, obj_class4, obj_class5))
    result = cursor.fetchall()
    if not result:
        return {"object_class_ids": []}
    return {"object_class_ids": [row[0] for row in result]}

# Endpoint to get attribute class ID based on attribute class
@app.get("/v1/image_and_language/attribute_class_id", operation_id="get_attribute_class_id", summary="Retrieves the unique identifier (ID) of an attribute class based on the provided attribute class name. This operation allows you to look up the ID associated with a specific attribute class, facilitating further data retrieval or manipulation.")
async def get_attribute_class_id(att_class: str = Query(..., description="Attribute class")):
    cursor.execute("SELECT ATT_CLASS_ID FROM ATT_CLASSES WHERE ATT_CLASS = ?", (att_class,))
    result = cursor.fetchone()
    if not result:
        return {"attribute_class_id": []}
    return {"attribute_class_id": result[0]}

# Endpoint to get object class ID based on object class
@app.get("/v1/image_and_language/object_class_id", operation_id="get_object_class_id", summary="Retrieves the unique identifier (ID) of a specific object class. The operation requires the object class as input and returns the corresponding ID from the database.")
async def get_object_class_id(obj_class: str = Query(..., description="Object class")):
    cursor.execute("SELECT OBJ_CLASS_ID FROM OBJ_CLASSES WHERE OBJ_CLASS = ?", (obj_class,))
    result = cursor.fetchone()
    if not result:
        return {"object_class_id": []}
    return {"object_class_id": result[0]}

# Endpoint to get attribute class based on image ID
@app.get("/v1/image_and_language/attribute_class_by_image_id", operation_id="get_attribute_class_by_image_id", summary="Get attribute class based on image ID")
async def get_attribute_class_by_image_id(img_id: int = Query(..., description="Image ID")):
    cursor.execute("SELECT T2.ATT_CLASS FROM IMG_OBJ_att AS T1 INNER JOIN ATT_CLASSES AS T2 ON T1.ATT_CLASS_ID = T2.ATT_CLASS_ID WHERE T1.IMG_ID = ?", (img_id,))
    result = cursor.fetchone()
    if not result:
        return {"attribute_class": []}
    return {"attribute_class": result[0]}

# Endpoint to get the count of images with a specific attribute class and minimum number of attributes
@app.get("/v1/image_and_language/count_images_with_attributes", operation_id="get_count_images_with_attributes", summary="Retrieves the total count of images that belong to a specified attribute class and possess at least a minimum number of attributes. The attribute class is defined by the 'att_class' parameter, while the minimum attribute count is set using the 'min_attributes' parameter.")
async def get_count_images_with_attributes(att_class: str = Query(..., description="Attribute class"), min_attributes: int = Query(..., description="Minimum number of attributes")):
    cursor.execute("SELECT COUNT(IMGID) FROM ( SELECT T1.IMG_ID AS IMGID FROM IMG_OBJ_att AS T1 INNER JOIN ATT_CLASSES AS T2 ON T1.ATT_CLASS_ID = T2.ATT_CLASS_ID WHERE T2.ATT_CLASS = ? GROUP BY T1.IMG_ID HAVING COUNT(T1.ATT_CLASS_ID) >= ? ) T3", (att_class, min_attributes))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the prediction class ID of the object with the highest height
@app.get("/v1/image_and_language/pred_class_id_highest_height", operation_id="get_pred_class_id_highest_height", summary="Retrieves the prediction class ID of the object with the highest vertical dimension in the image. This operation identifies the object with the greatest height in the image and returns its associated prediction class ID. The prediction class ID corresponds to the object's predicted category, as determined by the image analysis model.")
async def get_pred_class_id_highest_height():
    cursor.execute("SELECT T1.PRED_CLASS_ID FROM IMG_REL AS T1 INNER JOIN IMG_OBJ AS T2 ON T1.IMG_ID = T2.IMG_ID ORDER BY T2.H DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"pred_class_id": []}
    return {"pred_class_id": result[0]}

# Endpoint to get the image ID with the most occurrences of a specific attribute class
@app.get("/v1/image_and_language/image_id_most_attributes", operation_id="get_image_id_most_attributes", summary="Retrieves the image ID that has the highest frequency of a specified attribute class. The attribute class is provided as an input parameter.")
async def get_image_id_most_attributes(att_class: str = Query(..., description="Attribute class")):
    cursor.execute("SELECT T1.IMG_ID AS IMGID FROM IMG_OBJ_att AS T1 INNER JOIN ATT_CLASSES AS T2 ON T1.ATT_CLASS_ID = T2.ATT_CLASS_ID WHERE T2.ATT_CLASS = ? GROUP BY T1.IMG_ID ORDER BY COUNT(T1.ATT_CLASS_ID) DESC LIMIT 1", (att_class,))
    result = cursor.fetchone()
    if not result:
        return {"image_id": []}
    return {"image_id": result[0]}

# Endpoint to get the count of prediction class IDs based on image ID and prediction class
@app.get("/v1/image_and_language/count_pred_class_ids", operation_id="get_count_pred_class_ids", summary="Retrieves the total count of prediction class IDs associated with a specific image and prediction class. The image is identified by its unique ID, and the prediction class is specified by its name. This operation does not return the prediction class IDs themselves, but rather the total number of unique IDs that meet the specified criteria.")
async def get_count_pred_class_ids(img_id: int = Query(..., description="Image ID"), pred_class: str = Query(..., description="Prediction class")):
    cursor.execute("SELECT COUNT(T2.PRED_CLASS_ID) FROM IMG_REL AS T1 INNER JOIN PRED_CLASSES AS T2 ON T1.PRED_CLASS_ID = T2.PRED_CLASS_ID WHERE T1.IMG_ID = ? AND T2.PRED_CLASS = ?", (img_id, pred_class))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get object classes based on coordinates
@app.get("/v1/image_and_language/object_classes_by_coordinates", operation_id="get_object_classes_by_coordinates", summary="Retrieves the distinct object classes associated with the provided X and Y coordinates. The operation returns a list of object classes that are found at the specified location in the image.")
async def get_object_classes_by_coordinates(x: int = Query(..., description="X coordinate"), y: int = Query(..., description="Y coordinate")):
    cursor.execute("SELECT T1.OBJ_CLASS FROM OBJ_CLASSES AS T1 INNER JOIN IMG_OBJ AS T2 ON T1.OBJ_CLASS_ID = T2.OBJ_CLASS_ID WHERE T2.X = ? AND T2.Y = ? GROUP BY T1.OBJ_CLASS", (x, y))
    result = cursor.fetchall()
    if not result:
        return {"object_classes": []}
    return {"object_classes": result}

# Endpoint to get prediction classes where object sample IDs are the same
@app.get("/v1/image_and_language/pred_classes_same_sample_ids", operation_id="get_pred_classes_same_sample_ids", summary="Retrieves the distinct prediction classes for image pairs that share the same object sample IDs. This operation returns a list of prediction classes that are common to image pairs with matching object sample IDs.")
async def get_pred_classes_same_sample_ids():
    cursor.execute("SELECT T2.PRED_CLASS FROM IMG_REL AS T1 INNER JOIN pred_classes AS T2 ON T1.PRED_CLASS_ID = T2.PRED_CLASS_ID WHERE T1.OBJ1_SAMPLE_ID = T1.OBJ2_SAMPLE_ID GROUP BY T2.PRED_CLASS")
    result = cursor.fetchall()
    if not result:
        return {"pred_classes": []}
    return {"pred_classes": result}

# Endpoint to get object dimensions based on image ID and object class
@app.get("/v1/image_and_language/object_dimensions_by_image_and_class", operation_id="get_object_dimensions_by_image_and_class", summary="Retrieve the X, Y, height, and width dimensions of objects in a specific image, filtered by a given object class. The image is identified by its unique ID, and the object class is specified by its name.")
async def get_object_dimensions_by_image_and_class(img_id: int = Query(..., description="Image ID"), obj_class: str = Query(..., description="Object class")):
    cursor.execute("SELECT T2.X, T2.Y, T2.H, T2.W FROM OBJ_CLASSES AS T1 INNER JOIN IMG_OBJ AS T2 ON T1.OBJ_CLASS_ID = T2.OBJ_CLASS_ID WHERE T2.IMG_ID = ? AND T1.OBJ_CLASS = ?", (img_id, obj_class))
    result = cursor.fetchall()
    if not result:
        return {"dimensions": []}
    return {"dimensions": result}

# Endpoint to get the count of prediction class IDs where object sample IDs are different and prediction class is specified
@app.get("/v1/image_and_language/count_pred_class_ids_different_sample_ids", operation_id="get_count_pred_class_ids_different_sample_ids", summary="Retrieves the count of unique prediction class IDs where the object sample IDs are different and the prediction class matches the provided input. This operation is useful for understanding the distribution of prediction classes across different object samples.")
async def get_count_pred_class_ids_different_sample_ids(pred_class: str = Query(..., description="Prediction class")):
    cursor.execute("SELECT COUNT(T2.PRED_CLASS_ID) FROM IMG_REL AS T1 INNER JOIN PRED_CLASSES AS T2 ON T1.PRED_CLASS_ID = T2.PRED_CLASS_ID WHERE T1.OBJ1_SAMPLE_ID != T1.OBJ2_SAMPLE_ID AND T2.PRED_CLASS = ?", (pred_class,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get object classes based on image object dimensions and position
@app.get("/v1/image_and_language/obj_classes_by_dimensions_position", operation_id="get_obj_classes_by_dimensions_position", summary="Retrieves the classes of objects that match the specified position and dimensions within an image. The position is determined by the X and Y coordinates, while the dimensions are defined by the width and height. This operation returns the class of each object that meets these criteria.")
async def get_obj_classes_by_dimensions_position(x: int = Query(..., description="X coordinate of the image object"), y: int = Query(..., description="Y coordinate of the image object"), w: int = Query(..., description="Width of the image object"), h: int = Query(..., description="Height of the image object")):
    cursor.execute("SELECT T1.OBJ_CLASS FROM OBJ_CLASSES AS T1 INNER JOIN IMG_OBJ AS T2 ON T1.OBJ_CLASS_ID = T2.OBJ_CLASS_ID WHERE T2.X = ? AND T2.Y = ? AND T2.W = ? AND T2.H = ?", (x, y, w, h))
    result = cursor.fetchall()
    if not result:
        return {"obj_classes": []}
    return {"obj_classes": [row[0] for row in result]}

# Endpoint to get dimensions of image objects based on image ID and object class
@app.get("/v1/image_and_language/dimensions_by_img_id_obj_class", operation_id="get_dimensions_by_img_id_obj_class", summary="Retrieve the width and height dimensions of specific image objects based on the provided image ID and object class. This operation fetches the dimensions from the image objects table, which is joined with the object classes table using the object class ID. The result is a set of width and height values for the image objects that match the given image ID and object class.")
async def get_dimensions_by_img_id_obj_class(img_id: int = Query(..., description="Image ID"), obj_class: str = Query(..., description="Object class")):
    cursor.execute("SELECT T1.W, T1.H FROM IMG_OBJ AS T1 INNER JOIN OBJ_CLASSES AS T2 ON T1.OBJ_CLASS_ID = T2.OBJ_CLASS_ID WHERE T1.IMG_ID = ? AND T2.OBJ_CLASS = ?", (img_id, obj_class))
    result = cursor.fetchall()
    if not result:
        return {"dimensions": []}
    return {"dimensions": [{"width": row[0], "height": row[1]} for row in result]}

# Endpoint to get positions of image objects based on image ID and object class
@app.get("/v1/image_and_language/positions_by_img_id_obj_class", operation_id="get_positions_by_img_id_obj_class", summary="Retrieves the X and Y coordinates of objects within a specific image, filtered by the provided object class. The image is identified using its unique ID.")
async def get_positions_by_img_id_obj_class(img_id: int = Query(..., description="Image ID"), obj_class: str = Query(..., description="Object class")):
    cursor.execute("SELECT T1.X, T1.Y FROM IMG_OBJ AS T1 INNER JOIN OBJ_CLASSES AS T2 ON T1.OBJ_CLASS_ID = T2.OBJ_CLASS_ID WHERE T1.IMG_ID = ? AND T2.OBJ_CLASS = ?", (img_id, obj_class))
    result = cursor.fetchall()
    if not result:
        return {"positions": []}
    return {"positions": [{"x": row[0], "y": row[1]} for row in result]}

# Endpoint to get dimensions and positions of image objects based on image ID and object class
@app.get("/v1/image_and_language/dimensions_positions_by_img_id_obj_class", operation_id="get_dimensions_positions_by_img_id_obj_class", summary="Retrieves the dimensions (width and height) and positions (x and y coordinates) of specific objects within an image. The image is identified by its unique ID, and the objects are filtered by their class. This operation is useful for extracting object-level details from images, enabling further analysis or manipulation.")
async def get_dimensions_positions_by_img_id_obj_class(img_id: int = Query(..., description="Image ID"), obj_class: str = Query(..., description="Object class")):
    cursor.execute("SELECT T1.X, T1.Y, T1.W, T1.H FROM IMG_OBJ AS T1 INNER JOIN OBJ_CLASSES AS T2 ON T1.OBJ_CLASS_ID = T2.OBJ_CLASS_ID WHERE T1.IMG_ID = ? AND T2.OBJ_CLASS = ?", (img_id, obj_class))
    result = cursor.fetchall()
    if not result:
        return {"dimensions_positions": []}
    return {"dimensions_positions": [{"x": row[0], "y": row[1], "width": row[2], "height": row[3]} for row in result]}

# Endpoint to get sums based on image ID and object dimensions
@app.get("/v1/image_and_language/sums_by_img_id_dimensions", operation_id="get_sums_by_img_id_dimensions", summary="Retrieves the total count of objects and the count of objects with specific dimensions for a given image ID. The input parameters define the image ID and the dimensions (x, y, width, and height) of the objects to be considered for the count.")
async def get_sums_by_img_id_dimensions(img_id: int = Query(..., description="Image ID"), x: int = Query(..., description="X coordinate of the image object"), y: int = Query(..., description="Y coordinate of the image object"), w: int = Query(..., description="Width of the image object"), h: int = Query(..., description="Height of the image object")):
    cursor.execute("SELECT SUM(IIF(T1.IMG_ID = ?, 1, 0)), SUM(IIF(T1.X = ? AND T1.Y = ? AND T1.W = ? AND T1.H = ?, 1, 0)) FROM IMG_OBJ AS T1 INNER JOIN OBJ_CLASSES AS T2 ON T1.OBJ_CLASS_ID = T2.OBJ_CLASS_ID", (img_id, x, y, w, h))
    result = cursor.fetchone()
    if not result:
        return {"sums": []}
    return {"sums": {"img_id_sum": result[0], "dimensions_sum": result[1]}}

# Endpoint to get attribute classes based on image ID and count threshold
@app.get("/v1/image_and_language/att_classes_by_img_id_count_threshold", operation_id="get_att_classes_by_img_id_count_threshold", summary="Retrieve the attribute classes associated with a specific image ID that have a count greater than the provided threshold. This operation filters the attribute classes based on the given image ID and count threshold, returning only those that meet the specified criteria.")
async def get_att_classes_by_img_id_count_threshold(img_id: int = Query(..., description="Image ID"), count_threshold: int = Query(..., description="Count threshold")):
    cursor.execute("SELECT T2.ATT_CLASS FROM IMG_OBJ_ATT AS T1 INNER JOIN ATT_CLASSES AS T2 ON T1.ATT_CLASS_ID = T2.ATT_CLASS_ID WHERE T1.IMG_ID = ? GROUP BY T2.ATT_CLASS HAVING COUNT(T2.ATT_CLASS) > ?", (img_id, count_threshold))
    result = cursor.fetchall()
    if not result:
        return {"att_classes": []}
    return {"att_classes": [row[0] for row in result]}

# Endpoint to get attribute classes based on object class and image ID
@app.get("/v1/image_and_language/att_classes_by_obj_class_img_id", operation_id="get_att_classes_by_obj_class_img_id", summary="Retrieves the distinct attribute classes associated with a specific object class and image ID. The operation filters the results based on the provided object class and image ID, returning a list of unique attribute classes linked to the object class within the specified image.")
async def get_att_classes_by_obj_class_img_id(obj_class: str = Query(..., description="Object class"), img_id: int = Query(..., description="Image ID")):
    cursor.execute("SELECT T4.ATT_CLASS FROM IMG_OBJ AS T1 INNER JOIN OBJ_CLASSES AS T2 ON T1.OBJ_CLASS_ID = T2.OBJ_CLASS_ID INNER JOIN IMG_OBJ_ATT AS T3 ON T1.IMG_ID = T3.IMG_ID INNER JOIN ATT_CLASSES AS T4 ON T3.ATT_CLASS_ID = T4.ATT_CLASS_ID WHERE T2.OBJ_CLASS = ? AND T1.IMG_ID = ? GROUP BY T4.ATT_CLASS", (obj_class, img_id))
    result = cursor.fetchall()
    if not result:
        return {"att_classes": []}
    return {"att_classes": [row[0] for row in result]}

# Endpoint to get object classes based on attribute class and image ID
@app.get("/v1/image_and_language/obj_classes_by_att_class_img_id", operation_id="get_obj_classes_by_att_class_img_id", summary="Retrieves the distinct object classes associated with a specific attribute class and image ID. The operation filters objects based on the provided attribute class and image ID, then groups the results by object class.")
async def get_obj_classes_by_att_class_img_id(att_class: str = Query(..., description="Attribute class"), img_id: int = Query(..., description="Image ID")):
    cursor.execute("SELECT T2.OBJ_CLASS FROM IMG_OBJ AS T1 INNER JOIN OBJ_CLASSES AS T2 ON T1.OBJ_CLASS_ID = T2.OBJ_CLASS_ID INNER JOIN IMG_OBJ_ATT AS T3 ON T1.IMG_ID = T3.IMG_ID INNER JOIN ATT_CLASSES AS T4 ON T3.ATT_CLASS_ID = T4.ATT_CLASS_ID WHERE T4.ATT_CLASS = ? AND T1.IMG_ID = ? GROUP BY T2.OBJ_CLASS", (att_class, img_id))
    result = cursor.fetchall()
    if not result:
        return {"obj_classes": []}
    return {"obj_classes": [row[0] for row in result]}

# Endpoint to get the count of distinct image IDs based on object class
@app.get("/v1/image_and_language/count_distinct_img_ids_by_obj_class", operation_id="get_count_distinct_img_ids_by_obj_class", summary="Retrieves the number of unique images associated with a specific object class. The object class is provided as an input parameter, and the operation returns the count of distinct image IDs that correspond to this class.")
async def get_count_distinct_img_ids_by_obj_class(obj_class: str = Query(..., description="Object class")):
    cursor.execute("SELECT COUNT(DISTINCT T1.IMG_ID) FROM IMG_OBJ AS T1 INNER JOIN OBJ_CLASSES AS T2 ON T1.OBJ_CLASS_ID = T2.OBJ_CLASS_ID WHERE T2.OBJ_CLASS = ?", (obj_class,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the average number of object classes per image
@app.get("/v1/image_and_language/avg_obj_classes_per_image", operation_id="get_avg_obj_classes_per_image", summary="Retrieves the average count of distinct object classes per image, calculated by dividing the total count of object classes by the number of unique images.")
async def get_avg_obj_classes_per_image():
    cursor.execute("SELECT CAST(COUNT(OBJ_CLASS_ID) AS REAL) / COUNT(DISTINCT IMG_ID) FROM IMG_OBJ")
    result = cursor.fetchone()
    if not result:
        return {"average": []}
    return {"average": result[0]}

# Endpoint to get distinct object classes, attribute classes, and predicate classes for a specific image ID and bounding box coordinates
@app.get("/v1/image_and_language/distinct_classes_by_image_and_bbox", operation_id="get_distinct_classes", summary="Retrieve unique object, attribute, and predicate classes associated with a specific image and bounding box. The image is identified by its unique ID, and the bounding box is defined by its coordinates and dimensions. This operation provides a comprehensive view of the distinct classes present in the image within the specified area.")
async def get_distinct_classes(img_id: int = Query(..., description="Image ID"), x: int = Query(..., description="X coordinate"), y: int = Query(..., description="Y coordinate"), w: int = Query(..., description="Width"), h: int = Query(..., description="Height")):
    cursor.execute("SELECT DISTINCT T2.OBJ_CLASS, T4.ATT_CLASS, T6.PRED_CLASS FROM IMG_OBJ AS T1 INNER JOIN OBJ_CLASSES AS T2 ON T1.OBJ_CLASS_ID = T2.OBJ_CLASS_ID INNER JOIN IMG_OBJ_ATT AS T3 ON T1.IMG_ID = T3.IMG_ID INNER JOIN ATT_CLASSES AS T4 ON T3.ATT_CLASS_ID = T4.ATT_CLASS_ID INNER JOIN IMG_REL AS T5 ON T1.IMG_ID = T5.IMG_ID INNER JOIN PRED_CLASSES AS T6 ON T5.PRED_CLASS_ID = T6.PRED_CLASS_ID WHERE T1.IMG_ID = ? AND T1.X = ? AND T1.Y = ? AND T1.W = ? AND T1.H = ?", (img_id, x, y, w, h))
    result = cursor.fetchall()
    if not result:
        return {"classes": []}
    return {"classes": [{"obj_class": row[0], "att_class": row[1], "pred_class": row[2]} for row in result]}

# Endpoint to get the count of attribute class IDs for a specific image ID and object sample ID
@app.get("/v1/image_and_language/count_attribute_class_ids", operation_id="get_count_attribute_class_ids", summary="Retrieves the total number of unique attribute class IDs associated with a specific image and object sample. The image is identified by its unique ID, and the object sample is identified by its unique ID. This operation does not return the attribute class IDs themselves, but rather the count of distinct IDs.")
async def get_count_attribute_class_ids(img_id: int = Query(..., description="Image ID"), obj_sample_id: int = Query(..., description="Object sample ID")):
    cursor.execute("SELECT COUNT(ATT_CLASS_ID) FROM IMG_OBJ_ATT WHERE IMG_ID = ? AND OBJ_SAMPLE_ID = ?", (img_id, obj_sample_id))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of object class IDs for a specific image ID
@app.get("/v1/image_and_language/count_object_class_ids", operation_id="get_count_object_class_ids", summary="Retrieves the total number of unique object class identifiers associated with a specific image. The image is identified by the provided image ID.")
async def get_count_object_class_ids(img_id: int = Query(..., description="Image ID")):
    cursor.execute("SELECT COUNT(OBJ_CLASS_ID) FROM IMG_OBJ WHERE IMG_ID = ?", (img_id,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the most frequent attribute class ID for a specific image ID
@app.get("/v1/image_and_language/most_frequent_attribute_class", operation_id="get_most_frequent_attribute_class", summary="Retrieves the attribute class ID that appears most frequently for a given image ID. This operation groups the attribute classes by their IDs and orders them by their frequency of occurrence in descending order. The attribute class ID with the highest frequency is then returned.")
async def get_most_frequent_attribute_class(img_id: int = Query(..., description="Image ID")):
    cursor.execute("SELECT ATT_CLASS_ID FROM IMG_OBJ_ATT WHERE IMG_ID = ? GROUP BY ATT_CLASS_ID ORDER BY COUNT(ATT_CLASS_ID) DESC LIMIT 1", (img_id,))
    result = cursor.fetchone()
    if not result:
        return {"attribute_class_id": []}
    return {"attribute_class_id": result[0]}

# Endpoint to get bounding box coordinates for a specific image ID and object sample ID
@app.get("/v1/image_and_language/bounding_box_coordinates", operation_id="get_bounding_box_coordinates", summary="Retrieves the bounding box coordinates (X, Y, W, H) for a specific image and object sample. The image and object sample are identified by their respective IDs, which are provided as input parameters.")
async def get_bounding_box_coordinates(img_id: int = Query(..., description="Image ID"), obj_sample_id: int = Query(..., description="Object sample ID")):
    cursor.execute("SELECT X, Y, W, H FROM IMG_OBJ WHERE IMG_ID = ? AND OBJ_SAMPLE_ID = ?", (img_id, obj_sample_id))
    result = cursor.fetchone()
    if not result:
        return {"bounding_box": []}
    return {"bounding_box": {"x": result[0], "y": result[1], "w": result[2], "h": result[3]}}

# Endpoint to get the percentage of a specific attribute class for a given image ID
@app.get("/v1/image_and_language/percentage_attribute_class", operation_id="get_percentage_attribute_class", summary="Retrieves the percentage of a specified attribute class associated with a given image. The calculation is based on the total count of samples for the image and the count of samples with the specified attribute class.")
async def get_percentage_attribute_class(att_class: str = Query(..., description="Attribute class"), img_id: int = Query(..., description="Image ID")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.ATT_CLASS = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(OBJ_SAMPLE_ID) FROM IMG_OBJ_ATT AS T1 INNER JOIN ATT_CLASSES AS T2 ON T1.ATT_CLASS_ID = T2.ATT_CLASS_ID WHERE T1.IMG_ID = ?", (att_class, img_id))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the count of attribute class IDs for a specific image ID
@app.get("/v1/image_and_language/count_attribute_class_ids_by_image", operation_id="get_count_attribute_class_ids_by_image", summary="Retrieves the total number of unique attribute class identifiers associated with a specific image. The image is identified by the provided image ID.")
async def get_count_attribute_class_ids_by_image(img_id: int = Query(..., description="Image ID")):
    cursor.execute("SELECT COUNT(ATT_CLASS_ID) FROM IMG_OBJ_ATT WHERE IMG_ID = ?", (img_id,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the object class for a specific object class ID
@app.get("/v1/image_and_language/object_class_by_id", operation_id="get_object_class_by_id", summary="Retrieves the object class associated with the provided object class ID. This operation allows you to look up the specific object class by its unique identifier.")
async def get_object_class_by_id(obj_class_id: int = Query(..., description="Object class ID")):
    cursor.execute("SELECT OBJ_CLASS FROM OBJ_CLASSES WHERE OBJ_CLASS_ID = ?", (obj_class_id,))
    result = cursor.fetchone()
    if not result:
        return {"object_class": []}
    return {"object_class": result[0]}

# Endpoint to get the object class based on image object coordinates and dimensions
@app.get("/v1/image_and_language/get_object_class_by_coordinates", operation_id="get_object_class_by_coordinates", summary="Retrieves the class of an image object based on its coordinates and dimensions. The operation uses the provided X and Y coordinates, along with the width and height, to identify the object class from the database.")
async def get_object_class_by_coordinates(x: int = Query(..., description="X coordinate of the image object"), y: int = Query(..., description="Y coordinate of the image object"), w: int = Query(..., description="Width of the image object"), h: int = Query(..., description="Height of the image object")):
    cursor.execute("SELECT T2.OBJ_CLASS FROM IMG_OBJ AS T1 INNER JOIN OBJ_CLASSES AS T2 ON T1.OBJ_CLASS_ID = T2.OBJ_CLASS_ID WHERE T1.X = ? AND T1.Y = ? AND T1.W = ? AND T1.H = ?", (x, y, w, h))
    result = cursor.fetchall()
    if not result:
        return {"object_classes": []}
    return {"object_classes": [row[0] for row in result]}

# Endpoint to get the predicted class based on the prediction class ID
@app.get("/v1/image_and_language/get_predicted_class_by_id", operation_id="get_predicted_class_by_id", summary="Retrieves the predicted class associated with the provided prediction class ID. This operation fetches the specific predicted class from the database using the given ID as a reference.")
async def get_predicted_class_by_id(pred_class_id: int = Query(..., description="Prediction class ID")):
    cursor.execute("SELECT PRED_CLASS FROM PRED_CLASSES WHERE PRED_CLASS_ID = ?", (pred_class_id,))
    result = cursor.fetchone()
    if not result:
        return {"predicted_class": []}
    return {"predicted_class": result[0]}

# Endpoint to get the coordinates and dimensions of an image object based on the image ID
@app.get("/v1/image_and_language/get_image_object_dimensions", operation_id="get_image_object_dimensions", summary="Retrieves the coordinates and dimensions of a specific image object, identified by its unique image ID. The operation returns the X and Y coordinates of the object's top-left corner, as well as its width (W) and height (H).")
async def get_image_object_dimensions(img_id: int = Query(..., description="Image ID")):
    cursor.execute("SELECT X, Y, W, H FROM IMG_OBJ WHERE IMG_ID = ?", (img_id,))
    result = cursor.fetchall()
    if not result:
        return {"dimensions": []}
    return {"dimensions": [{"x": row[0], "y": row[1], "w": row[2], "h": row[3]} for row in result]}

# Endpoint to get the count of a specific object class in a specific image
@app.get("/v1/image_and_language/count_object_class_in_specific_image", operation_id="count_object_class_in_specific_image", summary="Retrieves the total count of a specified object class within a particular image. The image is identified by its unique ID, and the object class is determined by its name. The operation returns the sum of instances of the specified object class found in the image.")
async def count_object_class_in_specific_image(img_id: int = Query(..., description="Image ID"), obj_class: str = Query(..., description="Object class")):
    cursor.execute("SELECT SUM(CASE WHEN T1.IMG_ID = ? THEN 1 ELSE 0 END) FROM IMG_OBJ AS T1 INNER JOIN OBJ_CLASSES AS T2 ON T1.OBJ_CLASS_ID = T2.OBJ_CLASS_ID WHERE T2.OBJ_CLASS = ?", (img_id, obj_class))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the percentage of a specific object class in an image
@app.get("/v1/image_and_language/get_object_class_percentage", operation_id="get_object_class_percentage", summary="Retrieves the percentage of a specified object class present in an image. The calculation is based on the total count of the object class in the image and the total number of object classes identified in the image. The image is identified by its unique ID, and the object class is specified by its class name.")
async def get_object_class_percentage(img_id: int = Query(..., description="Image ID"), obj_class: str = Query(..., description="Object class")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.OBJ_CLASS = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.OBJ_CLASS_ID) FROM IMG_OBJ AS T1 INNER JOIN OBJ_CLASSES AS T2 ON T1.OBJ_CLASS_ID = T2.OBJ_CLASS_ID WHERE T1.IMG_ID = ?", (obj_class, img_id))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the object class with the highest height
@app.get("/v1/image_and_language/highest_height_object_class", operation_id="get_highest_height_object_class", summary="Retrieves the object class with the maximum height from the image objects. This operation identifies the object class with the highest vertical dimension among all image objects and returns its name. The result provides insights into the tallest object class present in the images.")
async def get_highest_height_object_class():
    cursor.execute("SELECT T2.OBJ_CLASS FROM IMG_OBJ AS T1 INNER JOIN OBJ_CLASSES AS T2 ON T1.OBJ_CLASS_ID = T2.OBJ_CLASS_ID ORDER BY T1.H DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"object_class": []}
    return {"object_class": result[0]}

# Endpoint to get the percentage of a specific object class
@app.get("/v1/image_and_language/percentage_object_class", operation_id="get_percentage_object_class", summary="Retrieves the percentage of a specified object class from the total number of image objects. The calculation is based on the provided object class, which is matched against the object classes associated with the image objects.")
async def get_percentage_object_class(obj_class: str = Query(..., description="Object class to calculate the percentage for")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.OBJ_CLASS = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T2.OBJ_CLASS) FROM IMG_OBJ AS T1 INNER JOIN OBJ_CLASSES AS T2 ON T1.OBJ_CLASS_ID = T2.OBJ_CLASS_ID", (obj_class,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the count of images with a specific object class and image ID
@app.get("/v1/image_and_language/count_images_object_class_image_id", operation_id="get_count_images_object_class_image_id", summary="Retrieves the total number of images that match a specified object class and image ID. This operation is useful for determining the prevalence of a particular object class within a specific image.")
async def get_count_images_object_class_image_id(obj_class: str = Query(..., description="Object class to filter by"), img_id: int = Query(..., description="Image ID to filter by")):
    cursor.execute("SELECT COUNT(T1.IMG_ID) FROM IMG_OBJ AS T1 INNER JOIN OBJ_CLASSES AS T2 ON T1.OBJ_CLASS_ID = T2.OBJ_CLASS_ID WHERE T2.OBJ_CLASS = ? AND T1.IMG_ID = ?", (obj_class, img_id))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the object class with the lowest height
@app.get("/v1/image_and_language/lowest_height_object_class", operation_id="get_lowest_height_object_class", summary="Retrieves the object class with the lowest height from the available image objects. This operation identifies the object class with the smallest vertical dimension among all image objects, providing a concise summary of the smallest object class present in the image data.")
async def get_lowest_height_object_class():
    cursor.execute("SELECT T2.OBJ_CLASS FROM IMG_OBJ AS T1 INNER JOIN OBJ_CLASSES AS T2 ON T1.OBJ_CLASS_ID = T2.OBJ_CLASS_ID ORDER BY T1.H LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"object_class": []}
    return {"object_class": result[0]}

# Endpoint to get image IDs with more than a specified number of objects
@app.get("/v1/image_and_language/image_ids_with_object_count", operation_id="get_image_ids_with_object_count", summary="Retrieves image identifiers for images containing more than the specified minimum number of objects. This operation is useful for filtering images based on the number of objects they contain, which can be helpful in various image analysis tasks.")
async def get_image_ids_with_object_count(min_count: int = Query(..., description="Minimum number of objects in the image")):
    cursor.execute("SELECT IMG_ID FROM IMG_OBJ GROUP BY IMG_ID HAVING COUNT(IMG_ID) > ?", (min_count,))
    result = cursor.fetchall()
    if not result:
        return {"image_ids": []}
    return {"image_ids": [row[0] for row in result]}

# Endpoint to get the object sample ID with the highest width in a given image
@app.get("/v1/image_and_language/highest_width_object_sample_id", operation_id="get_highest_width_object_sample_id", summary="Retrieves the unique identifier of the object within the specified image that has the maximum width. The operation filters the objects based on the provided image ID and returns the object sample ID of the object with the highest width.")
async def get_highest_width_object_sample_id(img_id: int = Query(..., description="Image ID to filter by")):
    cursor.execute("SELECT OBJ_SAMPLE_ID FROM IMG_OBJ WHERE IMG_ID = ? ORDER BY W DESC LIMIT 1", (img_id,))
    result = cursor.fetchone()
    if not result:
        return {"object_sample_id": []}
    return {"object_sample_id": result[0]}

# Endpoint to get the object sample ID based on image ID, X, and Y coordinates
@app.get("/v1/image_and_language/object_sample_id_by_coordinates", operation_id="get_object_sample_id_by_coordinates", summary="Retrieves the unique identifier of an object sample associated with a specific image, given the X and Y coordinates of the object within the image. This operation allows for precise identification of an object sample based on its spatial location within the image.")
async def get_object_sample_id_by_coordinates(img_id: int = Query(..., description="Image ID to filter by"), x: int = Query(..., description="X coordinate to filter by"), y: int = Query(..., description="Y coordinate to filter by")):
    cursor.execute("SELECT OBJ_SAMPLE_ID FROM IMG_OBJ WHERE IMG_ID = ? AND X = ? AND Y = ?", (img_id, x, y))
    result = cursor.fetchone()
    if not result:
        return {"object_sample_id": []}
    return {"object_sample_id": result[0]}

# Endpoint to get the object sample ID with the highest number of attributes
@app.get("/v1/image_and_language/highest_attribute_count_object_sample_id", operation_id="get_highest_attribute_count_object_sample_id", summary="Retrieves the unique identifier of the object sample that has the most associated attributes. This operation returns the object sample ID with the highest count of attributes, providing a quick way to identify the most complex or detailed object sample in the dataset.")
async def get_highest_attribute_count_object_sample_id():
    cursor.execute("SELECT OBJ_SAMPLE_ID FROM IMG_OBJ_ATT GROUP BY OBJ_SAMPLE_ID ORDER BY COUNT(OBJ_SAMPLE_ID) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"object_sample_id": []}
    return {"object_sample_id": result[0]}

# Endpoint to get the ratio of two image IDs based on their occurrences
@app.get("/v1/image_and_language/ratio_of_image_ids", operation_id="get_ratio_of_image_ids", summary="Retrieves the ratio of occurrences between two specified image IDs in the image object database. The operation compares the frequency of the first image ID to the second image ID, providing a relative measure of their prevalence.")
async def get_ratio_of_image_ids(img_id_1: int = Query(..., description="First image ID"), img_id_2: int = Query(..., description="Second image ID")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN IMG_ID = ? THEN 1 ELSE 0 END) AS REAL) / SUM(CASE WHEN IMG_ID = ? THEN 1 ELSE 0 END) FROM IMG_OBJ", (img_id_1, img_id_2))
    result = cursor.fetchone()
    if not result:
        return {"ratio": []}
    return {"ratio": result[0]}

# Endpoint to get the ratio of object samples to distinct image IDs
@app.get("/v1/image_and_language/ratio_obj_samples_to_img_ids", operation_id="get_ratio_obj_samples_to_img_ids", summary="Retrieves the ratio of total object samples to the number of distinct image IDs. This operation calculates the ratio by dividing the total count of object samples by the count of unique image IDs in the database.")
async def get_ratio_obj_samples_to_img_ids():
    cursor.execute("SELECT CAST(COUNT(OBJ_SAMPLE_ID) AS REAL) / COUNT(DISTINCT IMG_ID) FROM IMG_OBJ")
    result = cursor.fetchone()
    if not result:
        return {"ratio": []}
    return {"ratio": result[0]}

# Endpoint to get distinct image IDs based on attribute class
@app.get("/v1/image_and_language/distinct_img_ids_by_att_class", operation_id="get_distinct_img_ids_by_att_class", summary="Retrieve a unique set of image identifiers that correspond to a specified attribute class. This operation filters images based on the provided attribute class, ensuring that only distinct image IDs are returned.")
async def get_distinct_img_ids_by_att_class(att_class: str = Query(..., description="Attribute class")):
    cursor.execute("SELECT DISTINCT T2.IMG_ID FROM ATT_CLASSES AS T1 INNER JOIN IMG_OBJ_ATT AS T2 ON T1.ATT_CLASS_ID = T2.ATT_CLASS_ID WHERE T1.ATT_CLASS = ?", (att_class,))
    result = cursor.fetchall()
    if not result:
        return {"img_ids": []}
    return {"img_ids": [row[0] for row in result]}

# Endpoint to get distinct object classes based on image ID
@app.get("/v1/image_and_language/distinct_obj_classes_by_img_id", operation_id="get_distinct_obj_classes_by_img_id", summary="Retrieve a unique set of object classes associated with a specific image. This operation fetches distinct object classes from the image-object mapping, based on the provided image ID.")
async def get_distinct_obj_classes_by_img_id(img_id: int = Query(..., description="Image ID")):
    cursor.execute("SELECT DISTINCT T2.OBJ_CLASS FROM IMG_OBJ AS T1 INNER JOIN OBJ_CLASSES AS T2 ON T1.OBJ_CLASS_ID = T2.OBJ_CLASS_ID WHERE T1.IMG_ID = ?", (img_id,))
    result = cursor.fetchall()
    if not result:
        return {"obj_classes": []}
    return {"obj_classes": [row[0] for row in result]}

# Endpoint to get attribute classes based on image ID and object class
@app.get("/v1/image_and_language/att_classes_by_img_id_and_obj_class", operation_id="get_att_classes_by_img_id_and_obj_class", summary="Retrieves the attribute classes associated with a specific image and object class. The operation requires an image ID and an object class as input parameters. It returns a list of attribute classes that match the provided image ID and object class.")
async def get_att_classes_by_img_id_and_obj_class(img_id: int = Query(..., description="Image ID"), obj_class: str = Query(..., description="Object class")):
    cursor.execute("SELECT T1.ATT_CLASS FROM ATT_CLASSES AS T1 INNER JOIN IMG_OBJ_ATT AS T2 ON T1.ATT_CLASS_ID = T2.ATT_CLASS_ID INNER JOIN IMG_OBJ AS T3 ON T2.IMG_ID = T3.IMG_ID INNER JOIN OBJ_CLASSES AS T4 ON T3.OBJ_CLASS_ID = T4.OBJ_CLASS_ID WHERE T3.IMG_ID = ? AND T4.OBJ_CLASS = ?", (img_id, obj_class))
    result = cursor.fetchall()
    if not result:
        return {"att_classes": []}
    return {"att_classes": [row[0] for row in result]}

# Endpoint to get distinct predicate classes based on image ID and object sample IDs
@app.get("/v1/image_and_language/distinct_pred_classes_by_img_id_and_obj_sample_ids", operation_id="get_distinct_pred_classes_by_img_id_and_obj_sample_ids", summary="Get distinct predicate classes based on a specific image ID and object sample IDs")
async def get_distinct_pred_classes_by_img_id_and_obj_sample_ids(img_id: int = Query(..., description="Image ID"), obj1_sample_id: int = Query(..., description="First object sample ID"), obj2_sample_id: int = Query(..., description="Second object sample ID")):
    cursor.execute("SELECT DISTINCT T2.PRED_CLASS FROM IMG_REL AS T1 INNER JOIN PRED_CLASSES AS T2 ON T2.PRED_CLASS_ID = T1.PRED_CLASS_ID INNER JOIN IMG_OBJ AS T3 ON T1.IMG_ID = T3.IMG_ID AND T1.OBJ1_SAMPLE_ID = T3.OBJ_SAMPLE_ID INNER JOIN OBJ_CLASSES AS T4 ON T3.OBJ_CLASS_ID = T4.OBJ_CLASS_ID WHERE T1.IMG_ID = ? AND (T1.OBJ1_SAMPLE_ID = ? OR T1.OBJ2_SAMPLE_ID = ?)", (img_id, obj1_sample_id, obj2_sample_id))
    result = cursor.fetchall()
    if not result:
        return {"pred_classes": []}
    return {"pred_classes": [row[0] for row in result]}

# Endpoint to get the count of object samples based on attribute class
@app.get("/v1/image_and_language/count_obj_samples_by_att_class", operation_id="get_count_obj_samples_by_att_class", summary="Retrieves the total number of object samples associated with a specified attribute class. The attribute class is used to filter the object samples and calculate the count.")
async def get_count_obj_samples_by_att_class(att_class: str = Query(..., description="Attribute class")):
    cursor.execute("SELECT COUNT(T2.OBJ_SAMPLE_ID) FROM ATT_CLASSES AS T1 INNER JOIN IMG_OBJ_ATT AS T2 ON T1.ATT_CLASS_ID = T2.ATT_CLASS_ID WHERE T1.ATT_CLASS = ?", (att_class,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the top attribute class based on image ID and width
@app.get("/v1/image_and_language/top_att_class_by_img_id_and_width", operation_id="get_top_att_class_by_img_id_and_width", summary="Retrieves the top attribute class associated with a given image ID, based on the width of the image. The image is identified by its unique ID, and the attribute class is determined by considering the width of the image in descending order.")
async def get_top_att_class_by_img_id_and_width(img_id: int = Query(..., description="Image ID")):
    cursor.execute("SELECT T1.ATT_CLASS FROM ATT_CLASSES AS T1 INNER JOIN IMG_OBJ_ATT AS T2 ON T1.ATT_CLASS_ID = T2.ATT_CLASS_ID INNER JOIN IMG_OBJ AS T3 ON T2.IMG_ID = T3.IMG_ID WHERE T2.IMG_ID = ? ORDER BY T3.W DESC LIMIT 1", (img_id,))
    result = cursor.fetchone()
    if not result:
        return {"att_class": []}
    return {"att_class": result[0]}

# Endpoint to get the most common object class
@app.get("/v1/image_and_language/most_common_obj_class", operation_id="get_most_common_obj_class", summary="Retrieves the object class that appears most frequently in the image object dataset. The operation calculates the frequency of each object class in the dataset and returns the one with the highest count.")
async def get_most_common_obj_class():
    cursor.execute("SELECT OBJ_CLASS FROM IMG_OBJ AS T1 INNER JOIN OBJ_CLASSES AS T2 ON T1.OBJ_CLASS_ID = T2.OBJ_CLASS_ID GROUP BY T2.OBJ_CLASS ORDER BY COUNT(T1.OBJ_CLASS_ID) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"obj_class": []}
    return {"obj_class": result[0]}

# Endpoint to get height and width of objects based on image ID and object class
@app.get("/v1/image_and_language/height_width_by_img_id_and_obj_class", operation_id="get_height_width_by_img_id_and_obj_class", summary="Retrieves the height and width dimensions of a specific object class within a given image. The operation requires the image ID and the object class as input parameters to accurately locate and return the requested data.")
async def get_height_width_by_img_id_and_obj_class(img_id: int = Query(..., description="Image ID"), obj_class: str = Query(..., description="Object class")):
    cursor.execute("SELECT T1.H, T1.W FROM IMG_OBJ AS T1 INNER JOIN OBJ_CLASSES AS T2 ON T1.OBJ_CLASS_ID = T2.OBJ_CLASS_ID WHERE T1.IMG_ID = ? AND T2.OBJ_CLASS = ?", (img_id, obj_class))
    result = cursor.fetchall()
    if not result:
        return {"dimensions": []}
    return {"dimensions": [{"height": row[0], "width": row[1]} for row in result]}

# Endpoint to get object sample IDs, X, and Y coordinates based on image ID and attribute class
@app.get("/v1/image_and_language/obj_sample_id_x_y_by_img_id_att_class", operation_id="get_obj_sample_id_x_y", summary="Retrieves the object sample IDs, X and Y coordinates for a specific image based on the provided image ID and attribute class. The image ID is used to identify the image, while the attribute class filters the results to a specific category.")
async def get_obj_sample_id_x_y(img_id: int = Query(..., description="Image ID"), att_class: str = Query(..., description="Attribute class")):
    cursor.execute("SELECT T3.OBJ_SAMPLE_ID, T3.X, T3.Y FROM ATT_CLASSES AS T1 INNER JOIN IMG_OBJ_ATT AS T2 ON T1.ATT_CLASS_ID = T2.ATT_CLASS_ID INNER JOIN IMG_OBJ AS T3 ON T2.IMG_ID = T3.IMG_ID WHERE T3.IMG_ID = ? AND T1.ATT_CLASS = ?", (img_id, att_class))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get the percentage of objects of a specific class
@app.get("/v1/image_and_language/percentage_of_obj_class", operation_id="get_percentage_of_obj_class", summary="Retrieves the percentage of a specific object class from the total number of object samples. The object class is specified as an input parameter.")
async def get_percentage_of_obj_class(obj_class: str = Query(..., description="Object class")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.OBJ_CLASS = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.OBJ_SAMPLE_ID) FROM IMG_OBJ AS T1 INNER JOIN OBJ_CLASSES AS T2 ON T1.OBJ_CLASS_ID = T2.OBJ_CLASS_ID", (obj_class,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage of object samples in a specific image ID and object class
@app.get("/v1/image_and_language/percentage_of_obj_samples_by_img_id_obj_class", operation_id="get_percentage_of_obj_samples", summary="Retrieve the proportion of object samples associated with a specific image ID and object class. This operation calculates the ratio of the total count of object samples for the given object class to the total count of object samples in the specified image. The result is expressed as a percentage.")
async def get_percentage_of_obj_samples(img_id: int = Query(..., description="Image ID"), obj_class: str = Query(..., description="Object class")):
    cursor.execute("SELECT CAST(COUNT(T1.OBJ_SAMPLE_ID) AS REAL) * 100 / COUNT(CASE WHEN T1.IMG_ID = ? THEN 1 ELSE 0 END) FROM IMG_OBJ AS T1 INNER JOIN OBJ_CLASSES AS T2 ON T1.OBJ_CLASS_ID = T2.OBJ_CLASS_ID WHERE T2.OBJ_CLASS = ?", (img_id, obj_class))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the count of image IDs based on X and Y coordinates
@app.get("/v1/image_and_language/count_img_ids_by_x_y", operation_id="get_count_img_ids_by_x_y", summary="Retrieves the total number of unique image identifiers associated with the specified X and Y coordinates. This operation provides a count of images that share the same spatial location, enabling users to understand the distribution of images across a two-dimensional space.")
async def get_count_img_ids_by_x_y(x: int = Query(..., description="X coordinate"), y: int = Query(..., description="Y coordinate")):
    cursor.execute("SELECT COUNT(IMG_ID) FROM IMG_OBJ WHERE X = ? AND Y = ?", (x, y))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of image IDs based on object sample ID
@app.get("/v1/image_and_language/count_img_ids_by_obj_sample_id", operation_id="get_count_img_ids_by_obj_sample_id", summary="Retrieves the total number of unique image identifiers associated with an object sample ID that is less than the provided value. This operation is useful for determining the quantity of images linked to a specific object sample.")
async def get_count_img_ids_by_obj_sample_id(obj_sample_id: int = Query(..., description="Object sample ID")):
    cursor.execute("SELECT COUNT(IMG_ID) FROM IMG_OBJ WHERE OBJ_SAMPLE_ID < ?", (obj_sample_id,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of image IDs based on object class ID
@app.get("/v1/image_and_language/count_img_ids_by_obj_class_id", operation_id="get_count_img_ids_by_obj_class_id", summary="Retrieves the total number of unique image identifiers associated with a specific object class. The object class is identified by the provided object class ID.")
async def get_count_img_ids_by_obj_class_id(obj_class_id: int = Query(..., description="Object class ID")):
    cursor.execute("SELECT COUNT(IMG_ID) FROM IMG_OBJ WHERE OBJ_CLASS_ID = ?", (obj_class_id,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get image IDs based on distinct object sample IDs
@app.get("/v1/image_and_language/img_ids_by_distinct_obj_sample_ids", operation_id="get_img_ids_by_distinct_obj_sample_ids", summary="Retrieves a list of image IDs that contain at least one distinct object sample ID for each object class. This operation groups images by their object class and filters out those that do not have any distinct object sample IDs for both objects.")
async def get_img_ids_by_distinct_obj_sample_ids():
    cursor.execute("SELECT IMG_ID FROM IMG_REL GROUP BY PRED_CLASS_ID HAVING COUNT(DISTINCT OBJ1_SAMPLE_ID) != 0 AND COUNT(DISTINCT OBJ2_SAMPLE_ID) != 0")
    result = cursor.fetchall()
    if not result:
        return {"img_ids": []}
    return {"img_ids": [row[0] for row in result]}

# Endpoint to get the count of image IDs based on multiple object classes
@app.get("/v1/image_and_language/count_img_ids_by_obj_classes", operation_id="get_count_img_ids_by_obj_classes", summary="Retrieves the total count of unique image IDs that contain either of the two specified object classes. The response is based on a comparison of the provided object classes with the object classes associated with each image in the database.")
async def get_count_img_ids_by_obj_classes(obj_class_1: str = Query(..., description="First object class"), obj_class_2: str = Query(..., description="Second object class")):
    cursor.execute("SELECT COUNT(T1.IMG_ID) FROM IMG_OBJ AS T1 INNER JOIN OBJ_CLASSES AS T2 ON T1.OBJ_CLASS_ID = T2.OBJ_CLASS_ID WHERE T2.OBJ_CLASS = ? OR T2.OBJ_CLASS = ?", (obj_class_1, obj_class_2))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct image IDs based on predicate class
@app.get("/v1/image_and_language/distinct_img_ids_by_pred_class", operation_id="get_distinct_img_ids_by_pred_class", summary="Retrieve a unique set of image identifiers that are associated with a specified predicate class. This operation filters the image-relation table based on the provided predicate class and returns the distinct image IDs that match the filter criteria.")
async def get_distinct_img_ids_by_pred_class(pred_class: str = Query(..., description="Predicate class")):
    cursor.execute("SELECT DISTINCT T1.IMG_ID FROM IMG_REL AS T1 INNER JOIN PRED_CLASSES AS T2 ON T1.PRED_CLASS_ID = T2.PRED_CLASS_ID WHERE T2.PRED_CLASS = ?", (pred_class,))
    result = cursor.fetchall()
    if not result:
        return {"img_ids": []}
    return {"img_ids": [row[0] for row in result]}

# Endpoint to get object classes based on X and Y coordinates
@app.get("/v1/image_and_language/obj_classes_by_x_y", operation_id="get_obj_classes_by_x_y", summary="Retrieves the object classes associated with the provided X and Y coordinates. This operation fetches the object class information from the database by matching the given coordinates with the corresponding entries in the image object table. The result is a list of object classes that intersect with the specified location.")
async def get_obj_classes_by_x_y(x: int = Query(..., description="X coordinate"), y: int = Query(..., description="Y coordinate")):
    cursor.execute("SELECT T2.OBJ_CLASS FROM IMG_OBJ AS T1 INNER JOIN OBJ_CLASSES AS T2 ON T1.OBJ_CLASS_ID = T2.OBJ_CLASS_ID WHERE T1.X = ? AND T1.Y = ?", (x, y))
    result = cursor.fetchall()
    if not result:
        return {"obj_classes": []}
    return {"obj_classes": [row[0] for row in result]}

# Endpoint to get the count of images based on object class
@app.get("/v1/image_and_language/count_images_by_object_class", operation_id="get_count_images_by_object_class", summary="Retrieve the total number of images associated with a specified object class. The operation filters images based on the provided object class and returns the count of matching images.")
async def get_count_images_by_object_class(obj_class: str = Query(..., description="Object class to filter images")):
    cursor.execute("SELECT COUNT(T1.IMG_ID) FROM IMG_OBJ AS T1 INNER JOIN OBJ_CLASSES AS T2 ON T1.OBJ_CLASS_ID = T2.OBJ_CLASS_ID WHERE T2.OBJ_CLASS = ?", (obj_class,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get image IDs based on attribute class
@app.get("/v1/image_and_language/get_image_ids_by_attribute_class", operation_id="get_image_ids_by_attribute_class", summary="Retrieves a list of image IDs that match a specified attribute class. The attribute class is used to filter the images and return only those that meet the specified criteria.")
async def get_image_ids_by_attribute_class(att_class: str = Query(..., description="Attribute class to filter images")):
    cursor.execute("SELECT T2.IMG_ID FROM ATT_CLASSES AS T1 INNER JOIN IMG_OBJ_ATT AS T2 ON T1.ATT_CLASS_ID = T2.ATT_CLASS_ID WHERE T1.ATT_CLASS = ?", (att_class,))
    result = cursor.fetchall()
    if not result:
        return {"image_ids": []}
    return {"image_ids": [row[0] for row in result]}

# Endpoint to get attribute classes based on image ID
@app.get("/v1/image_and_language/get_attribute_classes_by_image_id", operation_id="get_attribute_classes_by_image_id", summary="Retrieves the distinct attribute classes associated with a specific image, identified by its unique image ID. This operation allows you to obtain a comprehensive list of attribute classes linked to the image, providing valuable insights into its characteristics.")
async def get_attribute_classes_by_image_id(img_id: int = Query(..., description="Image ID to filter attribute classes")):
    cursor.execute("SELECT T1.ATT_CLASS FROM ATT_CLASSES AS T1 INNER JOIN IMG_OBJ_ATT AS T2 ON T1.ATT_CLASS_ID = T2.ATT_CLASS_ID WHERE T2.IMG_ID = ?", (img_id,))
    result = cursor.fetchall()
    if not result:
        return {"attribute_classes": []}
    return {"attribute_classes": [row[0] for row in result]}

# Endpoint to get the count of images based on predicate class
@app.get("/v1/image_and_language/count_images_by_predicate_class", operation_id="get_count_images_by_predicate_class", summary="Retrieves the total number of images that belong to a specified predicate class. The predicate class is used to filter the images and determine the count.")
async def get_count_images_by_predicate_class(pred_class: str = Query(..., description="Predicate class to filter images")):
    cursor.execute("SELECT COUNT(T1.IMG_ID) FROM IMG_REL AS T1 INNER JOIN PRED_CLASSES AS T2 ON T1.PRED_CLASS_ID = T2.PRED_CLASS_ID WHERE T2.PRED_CLASS = ?", (pred_class,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of images based on attribute class
@app.get("/v1/image_and_language/count_images_by_attribute_class", operation_id="get_count_images_by_attribute_class", summary="Retrieve the total number of images associated with a specified attribute class. The attribute class is used to filter the images and calculate the count.")
async def get_count_images_by_attribute_class(att_class: str = Query(..., description="Attribute class to filter images")):
    cursor.execute("SELECT COUNT(T2.IMG_ID) FROM ATT_CLASSES AS T1 INNER JOIN IMG_OBJ_ATT AS T2 ON T1.ATT_CLASS_ID = T2.ATT_CLASS_ID WHERE T1.ATT_CLASS = ?", (att_class,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of images based on attribute class and object class
@app.get("/v1/image_and_language/count_images_by_attribute_and_object_class", operation_id="get_count_images_by_attribute_and_object_class", summary="Retrieves the total number of images that match a specified attribute class and object class. The attribute class and object class parameters are used to filter the images.")
async def get_count_images_by_attribute_and_object_class(att_class: str = Query(..., description="Attribute class to filter images"), obj_class: str = Query(..., description="Object class to filter images")):
    cursor.execute("SELECT COUNT(T2.IMG_ID) FROM ATT_CLASSES AS T1 INNER JOIN IMG_OBJ_ATT AS T2 ON T1.ATT_CLASS_ID = T2.ATT_CLASS_ID INNER JOIN IMG_OBJ AS T3 ON T2.IMG_ID = T3.IMG_ID INNER JOIN OBJ_CLASSES AS T4 ON T3.OBJ_CLASS_ID = T4.OBJ_CLASS_ID WHERE T1.ATT_CLASS = ? AND T4.OBJ_CLASS = ?", (att_class, obj_class))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get attribute classes based on object coordinates
@app.get("/v1/image_and_language/get_attribute_classes_by_coordinates", operation_id="get_attribute_classes_by_coordinates", summary="Retrieves the attribute classes associated with an object located at the specified coordinates. The operation uses the provided X and Y coordinates to identify the object and returns the corresponding attribute classes.")
async def get_attribute_classes_by_coordinates(x: int = Query(..., description="X coordinate of the object"), y: int = Query(..., description="Y coordinate of the object")):
    cursor.execute("SELECT T1.ATT_CLASS FROM ATT_CLASSES AS T1 INNER JOIN IMG_OBJ_ATT AS T2 ON T1.ATT_CLASS_ID = T2.ATT_CLASS_ID INNER JOIN IMG_OBJ AS T3 ON T2.IMG_ID = T3.IMG_ID WHERE T3.X = ? AND T3.Y = ?", (x, y))
    result = cursor.fetchall()
    if not result:
        return {"attribute_classes": []}
    return {"attribute_classes": [row[0] for row in result]}

# Endpoint to get the average image ID based on object class
@app.get("/v1/image_and_language/average_image_id_by_object_class", operation_id="get_average_image_id_by_object_class", summary="Retrieves the average image ID from a collection of images that belong to a specified object class. The object class is used as a filter to determine the relevant images for calculating the average.")
async def get_average_image_id_by_object_class(obj_class: str = Query(..., description="Object class to filter images")):
    cursor.execute("SELECT AVG(T1.IMG_ID) FROM IMG_OBJ AS T1 INNER JOIN OBJ_CLASSES AS T2 ON T1.OBJ_CLASS_ID = T2.OBJ_CLASS_ID WHERE T2.OBJ_CLASS = ?", (obj_class,))
    result = cursor.fetchone()
    if not result:
        return {"average_image_id": []}
    return {"average_image_id": result[0]}

# Endpoint to get the ratio of 'man' objects to 'person' objects
@app.get("/v1/image_and_language/ratio_man_to_person", operation_id="get_ratio_man_to_person", summary="Retrieves the proportion of 'man' objects relative to 'person' objects. This operation calculates the ratio by comparing the count of 'man' objects to the count of 'person' objects in the database. The input parameters specify the object classes for 'man' and 'person'.")
async def get_ratio_man_to_person(obj_class_man: str = Query(..., description="Object class for 'man'"), obj_class_person: str = Query(..., description="Object class for 'person")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T2.OBJ_CLASS = ? THEN 1 ELSE 0 END) AS REAL) / COUNT(CASE WHEN T2.OBJ_CLASS = ? THEN 1 ELSE 0 END) FROM IMG_OBJ AS T1 INNER JOIN OBJ_CLASSES AS T2 ON T1.OBJ_CLASS_ID = T2.OBJ_CLASS_ID", (obj_class_man, obj_class_person))
    result = cursor.fetchone()
    if not result:
        return {"ratio": []}
    return {"ratio": result[0]}

# Endpoint to get the width and height of an object based on object sample ID
@app.get("/v1/image_and_language/obj_dimensions_by_sample_id", operation_id="get_obj_dimensions", summary="Get the width and height of an object based on object sample ID")
async def get_obj_dimensions(obj_sample_id: int = Query(..., description="Object sample ID")):
    cursor.execute("SELECT W, H FROM IMG_OBJ WHERE OBJ_SAMPLE_ID = ?", (obj_sample_id,))
    result = cursor.fetchone()
    if not result:
        return {"dimensions": []}
    return {"dimensions": {"width": result[0], "height": result[1]}}

# Endpoint to get the count of images based on image ID and Y coordinate
@app.get("/v1/image_and_language/img_count_by_img_id_and_y", operation_id="get_img_count", summary="Retrieves the total number of images that match the specified image ID and Y coordinate. The image ID and Y coordinate are provided as input parameters to filter the results.")
async def get_img_count(img_id: int = Query(..., description="Image ID"), y: int = Query(..., description="Y coordinate")):
    cursor.execute("SELECT COUNT(IMG_ID) FROM IMG_OBJ WHERE IMG_ID = ? AND Y = ?", (img_id, y))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get object classes based on image ID and coordinates
@app.get("/v1/image_and_language/obj_classes_by_img_id_and_coords", operation_id="get_obj_classes", summary="Retrieves the object classes associated with a specific image, based on the provided image ID and coordinates. The endpoint returns the object class names for the objects located at the given coordinates within the specified image.")
async def get_obj_classes(img_id: int = Query(..., description="Image ID"), x: int = Query(..., description="X coordinate"), y: int = Query(..., description="Y coordinate")):
    cursor.execute("SELECT T2.OBJ_CLASS FROM IMG_OBJ AS T1 INNER JOIN OBJ_CLASSES AS T2 ON T1.OBJ_CLASS_ID = T2.OBJ_CLASS_ID WHERE T1.IMG_ID = ? AND T1.X = ? AND T1.Y = ?", (img_id, x, y))
    result = cursor.fetchall()
    if not result:
        return {"obj_classes": []}
    return {"obj_classes": [row[0] for row in result]}

# Endpoint to get image IDs and coordinates for a specific object class
@app.get("/v1/image_and_language/img_ids_and_coords_by_obj_class", operation_id="get_img_ids_and_coords", summary="Retrieves the first ten image IDs and their corresponding coordinates for a specified object class. The object class is identified by its unique name.")
async def get_img_ids_and_coords(obj_class: str = Query(..., description="Object class")):
    cursor.execute("SELECT T1.IMG_ID, T1.X, T1.Y FROM IMG_OBJ AS T1 INNER JOIN OBJ_CLASSES AS T2 ON T1.OBJ_CLASS_ID = T2.OBJ_CLASS_ID WHERE T2.OBJ_CLASS = ? LIMIT 10", (obj_class,))
    result = cursor.fetchall()
    if not result:
        return {"img_ids_and_coords": []}
    return {"img_ids_and_coords": [{"img_id": row[0], "x": row[1], "y": row[2]} for row in result]}

# Endpoint to get image IDs and object classes based on coordinates
@app.get("/v1/image_and_language/img_ids_and_obj_classes_by_coords", operation_id="get_img_ids_and_obj_classes", summary="Retrieves the image IDs and corresponding object classes for the given X and Y coordinates. This operation uses the provided coordinates to filter the image objects and returns the associated image IDs and object classes.")
async def get_img_ids_and_obj_classes(x: int = Query(..., description="X coordinate"), y: int = Query(..., description="Y coordinate")):
    cursor.execute("SELECT T1.IMG_ID, T2.OBJ_CLASS FROM IMG_OBJ AS T1 INNER JOIN OBJ_CLASSES AS T2 ON T1.OBJ_CLASS_ID = T2.OBJ_CLASS_ID WHERE T1.X = ? AND T1.Y = ?", (x, y))
    result = cursor.fetchall()
    if not result:
        return {"img_ids_and_obj_classes": []}
    return {"img_ids_and_obj_classes": [{"img_id": row[0], "obj_class": row[1]} for row in result]}

# Endpoint to get dimensions and object classes based on image ID
@app.get("/v1/image_and_language/dimensions_and_obj_classes_by_img_id", operation_id="get_dimensions_and_obj_classes", summary="Retrieves the width, height, and object class associated with a specific image, identified by its unique image ID. This operation fetches the image dimensions and corresponding object class from the database, providing a comprehensive view of the image's properties.")
async def get_dimensions_and_obj_classes(img_id: int = Query(..., description="Image ID")):
    cursor.execute("SELECT T1.W, T1.H, T2.OBJ_CLASS FROM IMG_OBJ AS T1 INNER JOIN OBJ_CLASSES AS T2 ON T1.OBJ_CLASS_ID = T2.OBJ_CLASS_ID WHERE T1.IMG_ID = ?", (img_id,))
    result = cursor.fetchall()
    if not result:
        return {"dimensions_and_obj_classes": []}
    return {"dimensions_and_obj_classes": [{"width": row[0], "height": row[1], "obj_class": row[2]} for row in result]}

# Endpoint to get the predicted class for a given image ID
@app.get("/v1/image_and_language/predicted_class_by_image_id", operation_id="get_predicted_class", summary="Retrieves the predicted class associated with the specified image ID. This operation fetches the predicted class from the PRED_CLASSES table by joining it with the IMG_REL table using the PRED_CLASS_ID. The image ID is used to filter the results.")
async def get_predicted_class(img_id: int = Query(..., description="ID of the image")):
    cursor.execute("SELECT T2.PRED_CLASS FROM IMG_REL AS T1 INNER JOIN PRED_CLASSES AS T2 ON T1.PRED_CLASS_ID = T2.PRED_CLASS_ID WHERE T1.IMG_ID = ?", (img_id,))
    result = cursor.fetchall()
    if not result:
        return {"predicted_classes": []}
    return {"predicted_classes": [row[0] for row in result]}

# Endpoint to get the count of predicted classes for a given image ID and predicted class
@app.get("/v1/image_and_language/count_predicted_classes", operation_id="get_count_predicted_classes", summary="Retrieves the total count of instances for a specified predicted class associated with a given image ID. This operation provides a quantitative measure of the occurrences of a particular predicted class within the context of a specific image.")
async def get_count_predicted_classes(img_id: int = Query(..., description="ID of the image"), pred_class: str = Query(..., description="Predicted class")):
    cursor.execute("SELECT COUNT(T2.PRED_CLASS) FROM IMG_REL AS T1 INNER JOIN PRED_CLASSES AS T2 ON T1.PRED_CLASS_ID = T2.PRED_CLASS_ID WHERE T1.IMG_ID = ? AND T2.PRED_CLASS = ?", (img_id, pred_class))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the top predicted class for a given image ID
@app.get("/v1/image_and_language/top_predicted_class_by_image_id", operation_id="get_top_predicted_class", summary="Retrieves the top predicted class for a specific image, identified by its unique image ID. The prediction is based on the highest ranked class from the image-related data and the corresponding class data. The result is sorted in descending order and the top prediction is returned.")
async def get_top_predicted_class(img_id: int = Query(..., description="ID of the image")):
    cursor.execute("SELECT T2.PRED_CLASS FROM IMG_REL AS T1 INNER JOIN PRED_CLASSES AS T2 ON T1.PRED_CLASS_ID = T2.PRED_CLASS_ID WHERE T1.IMG_ID = ? ORDER BY T2.PRED_CLASS DESC LIMIT 1", (img_id,))
    result = cursor.fetchone()
    if not result:
        return {"top_predicted_class": []}
    return {"top_predicted_class": result[0]}

# Endpoint to get the count of attribute classes for a given image ID and attribute class
@app.get("/v1/image_and_language/count_attribute_classes", operation_id="get_count_attribute_classes", summary="Retrieves the total count of instances for a specified attribute class associated with a given image. The image is identified by its unique ID, and the attribute class is specified by its name. This operation provides a quantitative measure of the occurrences of a particular attribute class within the context of the specified image.")
async def get_count_attribute_classes(img_id: int = Query(..., description="ID of the image"), att_class: str = Query(..., description="Attribute class")):
    cursor.execute("SELECT COUNT(T1.ATT_CLASS) FROM ATT_CLASSES AS T1 INNER JOIN IMG_OBJ_ATT AS T2 ON T1.ATT_CLASS_ID = T2.ATT_CLASS_ID WHERE T2.IMG_ID = ? AND T1.ATT_CLASS = ?", (img_id, att_class))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the average width and height of objects for a given image ID grouped by object class
@app.get("/v1/image_and_language/average_object_dimensions", operation_id="get_average_object_dimensions", summary="Retrieves the average width and height of objects within a specific image, grouped by their respective classes. The operation requires the image's unique identifier to accurately fetch and compute the average dimensions.")
async def get_average_object_dimensions(img_id: int = Query(..., description="ID of the image")):
    cursor.execute("SELECT T2.OBJ_CLASS, AVG(T1.W), AVG(T1.H) FROM IMG_OBJ AS T1 INNER JOIN OBJ_CLASSES AS T2 ON T1.OBJ_CLASS_ID = T2.OBJ_CLASS_ID WHERE T1.IMG_ID = ? GROUP BY T2.OBJ_CLASS", (img_id,))
    result = cursor.fetchall()
    if not result:
        return {"average_dimensions": []}
    return {"average_dimensions": [{"obj_class": row[0], "avg_width": row[1], "avg_height": row[2]} for row in result]}

api_calls = [
    "/v1/image_and_language/count_object_samples_by_image_id?img_id=1",
    "/v1/image_and_language/count_images_with_object_samples_greater_than?min_object_samples=20",
    "/v1/image_and_language/image_with_most_object_samples",
    "/v1/image_and_language/object_sample_ids_by_image_and_class?img_id=1&obj_class_id=297",
    "/v1/image_and_language/count_image_relations_with_matching_object_samples?img_id=5",
    "/v1/image_and_language/object_coordinates_with_matching_object_samples?img_id=5",
    "/v1/image_and_language/count_object_class_in_image?obj_class=man&img_id=1",
    "/v1/image_and_language/count_images_with_object_class?obj_class=man",
    "/v1/image_and_language/object_classes_in_image?img_id=1",
    "/v1/image_and_language/predicate_classes_by_image_and_object_samples?img_id=1&obj1_sample_id=8&obj2_sample_id=4",
    "/v1/image_and_language/sum_pred_class_img_id_diff_obj_ids?pred_class=parked%20on&img_id=1",
    "/v1/image_and_language/pred_class_obj_sample_ids?obj1_sample_id=14&obj2_sample_id=14",
    "/v1/image_and_language/sum_pred_class_diff_obj_ids?pred_class=parked%20on",
    "/v1/image_and_language/img_ids_pred_class_diff_obj_ids_count?pred_class=parked%20on&count=2",
    "/v1/image_and_language/pred_class_img_id_same_obj_ids?img_id=5",
    "/v1/image_and_language/obj_coords_dimensions_img_id_pred_class?img_id=1&pred_class=by",
    "/v1/image_and_language/avg_y_coord_pred_class_img_id_diff_obj_ids?pred_class=parked%20on&img_id=1",
    "/v1/image_and_language/percentage_obj_class_img_id?obj_class=man&img_id=1",
    "/v1/image_and_language/count_att_classes",
    "/v1/image_and_language/count_obj_classes",
    "/v1/image_and_language/pred_class_count",
    "/v1/image_and_language/object_dimensions?img_id=2324765&obj_class=kite",
    "/v1/image_and_language/attribute_class_count?img_id=2347915&att_class=white",
    "/v1/image_and_language/obj_sample_id_by_image_pred_class_and_obj?img_id=2345524&pred_class=lying%20on&obj2_sample_id=1",
    "/v1/image_and_language/obj_sample_count_by_image_and_class?img_id=6&obj_class=food",
    "/v1/image_and_language/pred_class_by_object_classes_and_image?obj_class1=feathers&obj_class2=onion&img_id=2345528",
    "/v1/image_and_language/att_class_by_object_class_and_image?obj_class=weeds&img_id=2377988",
    "/v1/image_and_language/obj_class_by_att_class_and_image?att_class=blurry&img_id=22377993",
    "/v1/image_and_language/get_object_class?img_id=2320341&obj_sample_id=10",
    "/v1/image_and_language/get_object_class_ratio?obj_class1=broccoli&obj_class2=tomato",
    "/v1/image_and_language/count_images_with_attribute_classes?min_att_classes=25",
    "/v1/image_and_language/distinct_image_ids_same_object_sample",
    "/v1/image_and_language/image_with_most_attribute_classes",
    "/v1/image_and_language/object_class_ids?obj_class1=bus&obj_class2=train&obj_class3=aeroplane&obj_class4=car&obj_class5=etc",
    "/v1/image_and_language/attribute_class_id?att_class=very%20large",
    "/v1/image_and_language/object_class_id?obj_class=onion",
    "/v1/image_and_language/attribute_class_by_image_id?img_id=8",
    "/v1/image_and_language/count_images_with_attributes?att_class=black&min_attributes=5",
    "/v1/image_and_language/pred_class_id_highest_height",
    "/v1/image_and_language/image_id_most_attributes?att_class=white",
    "/v1/image_and_language/count_pred_class_ids?img_id=3050&pred_class=has",
    "/v1/image_and_language/object_classes_by_coordinates?x=0&y=0",
    "/v1/image_and_language/pred_classes_same_sample_ids",
    "/v1/image_and_language/object_dimensions_by_image_and_class?img_id=2222&obj_class=feathers",
    "/v1/image_and_language/count_pred_class_ids_different_sample_ids?pred_class=on",
    "/v1/image_and_language/obj_classes_by_dimensions_position?x=0&y=0&w=135&h=212",
    "/v1/image_and_language/dimensions_by_img_id_obj_class?img_id=3&obj_class=keyboard",
    "/v1/image_and_language/positions_by_img_id_obj_class?img_id=6&obj_class=folk",
    "/v1/image_and_language/dimensions_positions_by_img_id_obj_class?img_id=285930&obj_class=onion",
    "/v1/image_and_language/sums_by_img_id_dimensions?img_id=1&x=341&y=27&w=42&h=51",
    "/v1/image_and_language/att_classes_by_img_id_count_threshold?img_id=5&count_threshold=2",
    "/v1/image_and_language/att_classes_by_obj_class_img_id?obj_class=wall&img_id=27",
    "/v1/image_and_language/obj_classes_by_att_class_img_id?att_class=scattered&img_id=10",
    "/v1/image_and_language/count_distinct_img_ids_by_obj_class?obj_class=bridge",
    "/v1/image_and_language/avg_obj_classes_per_image",
    "/v1/image_and_language/distinct_classes_by_image_and_bbox?img_id=1&x=388&y=369&w=48&h=128",
    "/v1/image_and_language/count_attribute_class_ids?img_id=4&obj_sample_id=7",
    "/v1/image_and_language/count_object_class_ids?img_id=31",
    "/v1/image_and_language/most_frequent_attribute_class?img_id=20",
    "/v1/image_and_language/bounding_box_coordinates?img_id=42&obj_sample_id=7",
    "/v1/image_and_language/percentage_attribute_class?att_class=white&img_id=99",
    "/v1/image_and_language/count_attribute_class_ids_by_image?img_id=5",
    "/v1/image_and_language/object_class_by_id?obj_class_id=10",
    "/v1/image_and_language/get_object_class_by_coordinates?x=422&y=63&w=77&h=363",
    "/v1/image_and_language/get_predicted_class_by_id?pred_class_id=12",
    "/v1/image_and_language/get_image_object_dimensions?img_id=8",
    "/v1/image_and_language/count_object_class_in_specific_image?img_id=2315533&obj_class=clouds",
    "/v1/image_and_language/get_object_class_percentage?img_id=2654&obj_class=surface",
    "/v1/image_and_language/highest_height_object_class",
    "/v1/image_and_language/percentage_object_class?obj_class=airplane",
    "/v1/image_and_language/count_images_object_class_image_id?obj_class=animal&img_id=660",
    "/v1/image_and_language/lowest_height_object_class",
    "/v1/image_and_language/image_ids_with_object_count?min_count=20",
    "/v1/image_and_language/highest_width_object_sample_id?img_id=8",
    "/v1/image_and_language/object_sample_id_by_coordinates?img_id=5&x=634&y=468",
    "/v1/image_and_language/highest_attribute_count_object_sample_id",
    "/v1/image_and_language/ratio_of_image_ids?img_id_1=1&img_id_2=6",
    "/v1/image_and_language/ratio_obj_samples_to_img_ids",
    "/v1/image_and_language/distinct_img_ids_by_att_class?att_class=wired",
    "/v1/image_and_language/distinct_obj_classes_by_img_id?img_id=10",
    "/v1/image_and_language/att_classes_by_img_id_and_obj_class?img_id=1314&obj_class=tip",
    "/v1/image_and_language/distinct_pred_classes_by_img_id_and_obj_sample_ids?img_id=2360078&obj1_sample_id=15&obj2_sample_id=18",
    "/v1/image_and_language/count_obj_samples_by_att_class?att_class=polka%20dot",
    "/v1/image_and_language/top_att_class_by_img_id_and_width?img_id=400",
    "/v1/image_and_language/most_common_obj_class",
    "/v1/image_and_language/height_width_by_img_id_and_obj_class?img_id=1&obj_class=van",
    "/v1/image_and_language/obj_sample_id_x_y_by_img_id_att_class?img_id=1&att_class=sparse",
    "/v1/image_and_language/percentage_of_obj_class?obj_class=street%20lights",
    "/v1/image_and_language/percentage_of_obj_samples_by_img_id_obj_class?img_id=5&obj_class=windows",
    "/v1/image_and_language/count_img_ids_by_x_y?x=5&y=5",
    "/v1/image_and_language/count_img_ids_by_obj_sample_id?obj_sample_id=15",
    "/v1/image_and_language/count_img_ids_by_obj_class_id?obj_class_id=10",
    "/v1/image_and_language/img_ids_by_distinct_obj_sample_ids",
    "/v1/image_and_language/count_img_ids_by_obj_classes?obj_class_1=vegetables&obj_class_2=fruits",
    "/v1/image_and_language/distinct_img_ids_by_pred_class?pred_class=parked%20on",
    "/v1/image_and_language/obj_classes_by_x_y?x=5&y=5",
    "/v1/image_and_language/count_images_by_object_class?obj_class=keyboard",
    "/v1/image_and_language/get_image_ids_by_attribute_class?att_class=horse",
    "/v1/image_and_language/get_attribute_classes_by_image_id?img_id=15",
    "/v1/image_and_language/count_images_by_predicate_class?pred_class=reading",
    "/v1/image_and_language/count_images_by_attribute_class?att_class=picture",
    "/v1/image_and_language/count_images_by_attribute_and_object_class?att_class=picture&obj_class=bear",
    "/v1/image_and_language/get_attribute_classes_by_coordinates?x=5&y=5",
    "/v1/image_and_language/average_image_id_by_object_class?obj_class=keyboard",
    "/v1/image_and_language/ratio_man_to_person?obj_class_man=man&obj_class_person=person",
    "/v1/image_and_language/obj_dimensions_by_sample_id?obj_sample_id=2",
    "/v1/image_and_language/img_count_by_img_id_and_y?img_id=12&y=0",
    "/v1/image_and_language/obj_classes_by_img_id_and_coords?img_id=36&x=0&y=0",
    "/v1/image_and_language/img_ids_and_coords_by_obj_class?obj_class=pizza",
    "/v1/image_and_language/img_ids_and_obj_classes_by_coords?x=126&y=363",
    "/v1/image_and_language/dimensions_and_obj_classes_by_img_id?img_id=22",
    "/v1/image_and_language/predicted_class_by_image_id?img_id=68",
    "/v1/image_and_language/count_predicted_classes?img_id=107&pred_class=has",
    "/v1/image_and_language/top_predicted_class_by_image_id?img_id=4434",
    "/v1/image_and_language/count_attribute_classes?img_id=2355735&att_class=blue",
    "/v1/image_and_language/average_object_dimensions?img_id=47"
]
