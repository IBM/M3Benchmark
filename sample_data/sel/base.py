[
    {
        "uuid": "8a751d8b-4cdd-bdfe-54efdb1c4be8",
        "query": "SELECT COUNT(BrandID) FROM rootbeerbrand WHERE Country = 'United States'",
        "input": "How many breweries are located in North America?",
        "dataset_name": "beer_factory",
        "sample_id": 1,
        "gold_answer": 23,
        "output": [
            {
                "name": "initialize_active_data",
                "arguments": {
                    "condition_sequence": [],
                    "alias_to_table_dict": {
                        "": {
                            "original_table_name": "rootbeerbrand",
                            "modified_table_name": "rootbeerbrand"
                        }
                    },
                    "database_path": "/Users/belder/invocable-api-hub/db/cache/beer_factory/beer_factory.sqlite"
                },
                "label": "starting_table_var"
            },
            {
                "name": "select_data_equal_to",
                "arguments": {
                    "data_source": "$starting_table_var.data_path$",
                    "key_name": "Country",
                    "value": "United States"
                },
                "label": "FILTERED_DF_0"
            },
            {
                "name": "compute_data_count",
                "arguments": {
                    "data_source": "$FILTERED_DF_0.data_path$",
                    "key_name": "BrandID",
                    "distinct": false
                },
                "label": "COUNT"
            }
        ],
        "key_values_and_descriptions": [
            {
                "key_name": "BrandID",
                "description": "the unique id for the brand",
                "dtype": "integer"
            },
            {
                "key_name": "BrandName",
                "description": "the brand name",
                "dtype": "string"
            },
            {
                "key_name": "FirstBrewedYear",
                "description": "the first brewed year of the brand",
                "dtype": "integer"
            },
            {
                "key_name": "BreweryName",
                "description": "the brewery name",
                "dtype": "string"
            },
            {
                "key_name": "City",
                "description": "the city where the brewery locates",
                "dtype": "string"
            },
            {
                "key_name": "State",
                "description": "the state code",
                "dtype": "string"
            },
            {
                "key_name": "Country",
                "description": "the country where the brewery locates",
                "dtype": "string"
            },
            {
                "key_name": "Description",
                "description": "the description of the brand",
                "dtype": "string"
            },
            {
                "key_name": "CaneSugar",
                "description": "whether the drink contains cane sugar",
                "dtype": "string"
            },
            {
                "key_name": "CornSyrup",
                "description": "whether the drink contains the corn syrup",
                "dtype": "string"
            },
            {
                "key_name": "Honey",
                "description": "whether the drink contains the honey ",
                "dtype": "string"
            },
            {
                "key_name": "ArtificialSweetener",
                "description": "whether the drink contains the artificial sweetener ",
                "dtype": "string"
            },
            {
                "key_name": "Caffeinated",
                "description": "whether the drink is caffeinated",
                "dtype": "string"
            },
            {
                "key_name": "Alcoholic",
                "description": "whether the drink is alcoholic",
                "dtype": "string"
            },
            {
                "key_name": "AvailableInCans",
                "description": "whether the drink is available in cans",
                "dtype": "string"
            },
            {
                "key_name": "AvailableInBottles",
                "description": "whether the drink is available in bottles",
                "dtype": "string"
            },
            {
                "key_name": "AvailableInKegs",
                "description": "whether the drink is available in kegs",
                "dtype": "string"
            },
            {
                "key_name": "Website",
                "description": "the website of the brand",
                "dtype": "string"
            },
            {
                "key_name": "FacebookPage",
                "description": "the facebook page of the brand",
                "dtype": "string"
            },
            {
                "key_name": "Twitter",
                "description": "the twitter of the brand",
                "dtype": "string"
            },
            {
                "key_name": "WholesaleCost",
                "description": "the wholesale cost",
                "dtype": "number"
            },
            {
                "key_name": "CurrentRetailPrice",
                "description": "the current retail price",
                "dtype": "number"
            }
        ],
        "tools": [
            {
                "description": "Lookup data BrandID: the unique id for the brand",
                "name": "get_BrandIDs",
                "parameters": {
                    "properties": {
                        "data_source": {
                            "description": "The location of the data file in csv format.",
                            "schema": {
                                "type": "string"
                            }
                        }
                    },
                    "required": [
                        "data_source"
                    ],
                    "type": "object"
                },
                "output_parameters": {
                    "properties": {
                        "output_0": {
                            "description": "list of BrandIDs",
                            "type": "object"
                        }
                    }
                }
            },
            {
                "description": "Lookup data BrandName: the brand name",
                "name": "get_BrandNames",
                "parameters": {
                    "properties": {
                        "data_source": {
                            "description": "The location of the data file in csv format.",
                            "schema": {
                                "type": "string"
                            }
                        }
                    },
                    "required": [
                        "data_source"
                    ],
                    "type": "object"
                },
                "output_parameters": {
                    "properties": {
                        "output_0": {
                            "description": "list of BrandNames",
                            "type": "object"
                        }
                    }
                }
            },
            {
                "description": "Lookup data FirstBrewedYear: the first brewed year of the brand",
                "name": "get_FirstBrewedYears",
                "parameters": {
                    "properties": {
                        "data_source": {
                            "description": "The location of the data file in csv format.",
                            "schema": {
                                "type": "string"
                            }
                        }
                    },
                    "required": [
                        "data_source"
                    ],
                    "type": "object"
                },
                "output_parameters": {
                    "properties": {
                        "output_0": {
                            "description": "list of FirstBrewedYears",
                            "type": "object"
                        }
                    }
                }
            },
            {
                "description": "Lookup data BreweryName: the brewery name",
                "name": "get_BreweryNames",
                "parameters": {
                    "properties": {
                        "data_source": {
                            "description": "The location of the data file in csv format.",
                            "schema": {
                                "type": "string"
                            }
                        }
                    },
                    "required": [
                        "data_source"
                    ],
                    "type": "object"
                },
                "output_parameters": {
                    "properties": {
                        "output_0": {
                            "description": "list of BreweryNames",
                            "type": "object"
                        }
                    }
                }
            },
            {
                "description": "Lookup data City: the city where the brewery locates",
                "name": "get_Citys",
                "parameters": {
                    "properties": {
                        "data_source": {
                            "description": "The location of the data file in csv format.",
                            "schema": {
                                "type": "string"
                            }
                        }
                    },
                    "required": [
                        "data_source"
                    ],
                    "type": "object"
                },
                "output_parameters": {
                    "properties": {
                        "output_0": {
                            "description": "list of Citys",
                            "type": "object"
                        }
                    }
                }
            },
            {
                "description": "Lookup data State: the state code",
                "name": "get_States",
                "parameters": {
                    "properties": {
                        "data_source": {
                            "description": "The location of the data file in csv format.",
                            "schema": {
                                "type": "string"
                            }
                        }
                    },
                    "required": [
                        "data_source"
                    ],
                    "type": "object"
                },
                "output_parameters": {
                    "properties": {
                        "output_0": {
                            "description": "list of States",
                            "type": "object"
                        }
                    }
                }
            },
            {
                "description": "Lookup data Country: the country where the brewery locates",
                "name": "get_Countrys",
                "parameters": {
                    "properties": {
                        "data_source": {
                            "description": "The location of the data file in csv format.",
                            "schema": {
                                "type": "string"
                            }
                        }
                    },
                    "required": [
                        "data_source"
                    ],
                    "type": "object"
                },
                "output_parameters": {
                    "properties": {
                        "output_0": {
                            "description": "list of Countrys",
                            "type": "object"
                        }
                    }
                }
            },
            {
                "description": "Lookup data Description: the description of the brand",
                "name": "get_Descriptions",
                "parameters": {
                    "properties": {
                        "data_source": {
                            "description": "The location of the data file in csv format.",
                            "schema": {
                                "type": "string"
                            }
                        }
                    },
                    "required": [
                        "data_source"
                    ],
                    "type": "object"
                },
                "output_parameters": {
                    "properties": {
                        "output_0": {
                            "description": "list of Descriptions",
                            "type": "object"
                        }
                    }
                }
            },
            {
                "description": "Lookup data CaneSugar: whether the drink contains cane sugar",
                "name": "get_CaneSugars",
                "parameters": {
                    "properties": {
                        "data_source": {
                            "description": "The location of the data file in csv format.",
                            "schema": {
                                "type": "string"
                            }
                        }
                    },
                    "required": [
                        "data_source"
                    ],
                    "type": "object"
                },
                "output_parameters": {
                    "properties": {
                        "output_0": {
                            "description": "list of CaneSugars",
                            "type": "object"
                        }
                    }
                }
            },
            {
                "description": "Lookup data CornSyrup: whether the drink contains the corn syrup",
                "name": "get_CornSyrups",
                "parameters": {
                    "properties": {
                        "data_source": {
                            "description": "The location of the data file in csv format.",
                            "schema": {
                                "type": "string"
                            }
                        }
                    },
                    "required": [
                        "data_source"
                    ],
                    "type": "object"
                },
                "output_parameters": {
                    "properties": {
                        "output_0": {
                            "description": "list of CornSyrups",
                            "type": "object"
                        }
                    }
                }
            },
            {
                "description": "Lookup data Honey: whether the drink contains the honey",
                "name": "get_Honeys",
                "parameters": {
                    "properties": {
                        "data_source": {
                            "description": "The location of the data file in csv format.",
                            "schema": {
                                "type": "string"
                            }
                        }
                    },
                    "required": [
                        "data_source"
                    ],
                    "type": "object"
                },
                "output_parameters": {
                    "properties": {
                        "output_0": {
                            "description": "list of Honeys",
                            "type": "object"
                        }
                    }
                }
            },
            {
                "description": "Lookup data ArtificialSweetener: whether the drink contains the artificial sweetener",
                "name": "get_ArtificialSweeteners",
                "parameters": {
                    "properties": {
                        "data_source": {
                            "description": "The location of the data file in csv format.",
                            "schema": {
                                "type": "string"
                            }
                        }
                    },
                    "required": [
                        "data_source"
                    ],
                    "type": "object"
                },
                "output_parameters": {
                    "properties": {
                        "output_0": {
                            "description": "list of ArtificialSweeteners",
                            "type": "object"
                        }
                    }
                }
            },
            {
                "description": "Lookup data Caffeinated: whether the drink is caffeinated",
                "name": "get_Caffeinateds",
                "parameters": {
                    "properties": {
                        "data_source": {
                            "description": "The location of the data file in csv format.",
                            "schema": {
                                "type": "string"
                            }
                        }
                    },
                    "required": [
                        "data_source"
                    ],
                    "type": "object"
                },
                "output_parameters": {
                    "properties": {
                        "output_0": {
                            "description": "list of Caffeinateds",
                            "type": "object"
                        }
                    }
                }
            },
            {
                "description": "Lookup data Alcoholic: whether the drink is alcoholic",
                "name": "get_Alcoholics",
                "parameters": {
                    "properties": {
                        "data_source": {
                            "description": "The location of the data file in csv format.",
                            "schema": {
                                "type": "string"
                            }
                        }
                    },
                    "required": [
                        "data_source"
                    ],
                    "type": "object"
                },
                "output_parameters": {
                    "properties": {
                        "output_0": {
                            "description": "list of Alcoholics",
                            "type": "object"
                        }
                    }
                }
            },
            {
                "description": "Lookup data AvailableInCans: whether the drink is available in cans",
                "name": "get_AvailableInCanss",
                "parameters": {
                    "properties": {
                        "data_source": {
                            "description": "The location of the data file in csv format.",
                            "schema": {
                                "type": "string"
                            }
                        }
                    },
                    "required": [
                        "data_source"
                    ],
                    "type": "object"
                },
                "output_parameters": {
                    "properties": {
                        "output_0": {
                            "description": "list of AvailableInCanss",
                            "type": "object"
                        }
                    }
                }
            },
            {
                "description": "Lookup data AvailableInBottles: whether the drink is available in bottles",
                "name": "get_AvailableInBottless",
                "parameters": {
                    "properties": {
                        "data_source": {
                            "description": "The location of the data file in csv format.",
                            "schema": {
                                "type": "string"
                            }
                        }
                    },
                    "required": [
                        "data_source"
                    ],
                    "type": "object"
                },
                "output_parameters": {
                    "properties": {
                        "output_0": {
                            "description": "list of AvailableInBottless",
                            "type": "object"
                        }
                    }
                }
            },
            {
                "description": "Lookup data AvailableInKegs: whether the drink is available in kegs",
                "name": "get_AvailableInKegss",
                "parameters": {
                    "properties": {
                        "data_source": {
                            "description": "The location of the data file in csv format.",
                            "schema": {
                                "type": "string"
                            }
                        }
                    },
                    "required": [
                        "data_source"
                    ],
                    "type": "object"
                },
                "output_parameters": {
                    "properties": {
                        "output_0": {
                            "description": "list of AvailableInKegss",
                            "type": "object"
                        }
                    }
                }
            },
            {
                "description": "Lookup data Website: the website of the brand",
                "name": "get_Websites",
                "parameters": {
                    "properties": {
                        "data_source": {
                            "description": "The location of the data file in csv format.",
                            "schema": {
                                "type": "string"
                            }
                        }
                    },
                    "required": [
                        "data_source"
                    ],
                    "type": "object"
                },
                "output_parameters": {
                    "properties": {
                        "output_0": {
                            "description": "list of Websites",
                            "type": "object"
                        }
                    }
                }
            },
            {
                "description": "Lookup data FacebookPage: the facebook page of the brand",
                "name": "get_FacebookPages",
                "parameters": {
                    "properties": {
                        "data_source": {
                            "description": "The location of the data file in csv format.",
                            "schema": {
                                "type": "string"
                            }
                        }
                    },
                    "required": [
                        "data_source"
                    ],
                    "type": "object"
                },
                "output_parameters": {
                    "properties": {
                        "output_0": {
                            "description": "list of FacebookPages",
                            "type": "object"
                        }
                    }
                }
            },
            {
                "description": "Lookup data Twitter: the twitter of the brand",
                "name": "get_Twitters",
                "parameters": {
                    "properties": {
                        "data_source": {
                            "description": "The location of the data file in csv format.",
                            "schema": {
                                "type": "string"
                            }
                        }
                    },
                    "required": [
                        "data_source"
                    ],
                    "type": "object"
                },
                "output_parameters": {
                    "properties": {
                        "output_0": {
                            "description": "list of Twitters",
                            "type": "object"
                        }
                    }
                }
            },
            {
                "description": "Lookup data WholesaleCost: the wholesale cost",
                "name": "get_WholesaleCosts",
                "parameters": {
                    "properties": {
                        "data_source": {
                            "description": "The location of the data file in csv format.",
                            "schema": {
                                "type": "string"
                            }
                        }
                    },
                    "required": [
                        "data_source"
                    ],
                    "type": "object"
                },
                "output_parameters": {
                    "properties": {
                        "output_0": {
                            "description": "list of WholesaleCosts",
                            "type": "object"
                        }
                    }
                }
            },
            {
                "description": "Lookup data CurrentRetailPrice: the current retail price",
                "name": "get_CurrentRetailPrices",
                "parameters": {
                    "properties": {
                        "data_source": {
                            "description": "The location of the data file in csv format.",
                            "schema": {
                                "type": "string"
                            }
                        }
                    },
                    "required": [
                        "data_source"
                    ],
                    "type": "object"
                },
                "output_parameters": {
                    "properties": {
                        "output_0": {
                            "description": "list of CurrentRetailPrices",
                            "type": "object"
                        }
                    }
                }
            },
            {
                "description": "Concatenates two tables along axis 0 (rows)",
                "name": "concatenate_data",
                "parameters": {
                    "properties": {
                        "data_source_1": {
                            "description": "The location of the first data file in csv format.",
                            "schema": {
                                "type": "string"
                            }
                        },
                        "data_source_2": {
                            "description": "The location of the second data file in csv format.",
                            "schema": {
                                "type": "string"
                            }
                        }
                    },
                    "required": [
                        "data_source_1",
                        "data_source_2"
                    ],
                    "type": "object"
                },
                "output_parameters": {
                    "properties": {
                        "output_0": {
                            "type": "object",
                            "properties": {
                                "data_path": {
                                    "type": "string",
                                    "description": "location where full data can be accessed."
                                },
                                "num_records": {
                                    "type": "string",
                                    "description": "number of individual records"
                                },
                                "keys": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "key_details": {
                                    "type": "object",
                                    "properties": {
                                        "key1": {
                                            "type": "object",
                                            "properties": {
                                                "dtype": {
                                                    "type": "string",
                                                    "description": "\"int64\""
                                                },
                                                "first_3_values": {
                                                    "type": "array",
                                                    "items": {
                                                        "type": "string"
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            },
                            "description": "The following summary of the available data is returned:"
                        }
                    }
                }
            },
            {
                "description": "Initializes active data based on the provided condition sequence, alias-to-table mapping, and database path.  This function checks the validity of the database file at the specified path and raises an exception if the file is not found. After validating the database path, the function processes the condition sequence and alias-to-table dictionary to return a dictionary of active data.",
                "name": "initialize_active_data",
                "parameters": {
                    "properties": {
                        "condition_sequence": {
                            "description": "A list of conditions (joins) to be processed for initializing the data.",
                            "schema": {
                                "type": "array"
                            }
                        },
                        "alias_to_table_dict": {
                            "description": "A dictionary mapping aliases to their respective tables.",
                            "schema": {
                                "type": "object"
                            }
                        },
                        "database_path": {
                            "description": "The file path to the database that will be used for the initialization.",
                            "schema": {
                                "type": "string"
                            }
                        }
                    },
                    "required": [
                        "condition_sequence",
                        "alias_to_table_dict",
                        "database_path"
                    ],
                    "type": "object"
                },
                "output_parameters": {
                    "properties": {
                        "output_0": {
                            "type": "object",
                            "properties": {
                                "data_path": {
                                    "type": "string",
                                    "description": "location where full data can be accessed."
                                },
                                "num_records": {
                                    "type": "string",
                                    "description": "number of individual records"
                                },
                                "keys": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "key_details": {
                                    "type": "object",
                                    "properties": {
                                        "key1": {
                                            "type": "object",
                                            "properties": {
                                                "dtype": {
                                                    "type": "string",
                                                    "description": "\"int64\""
                                                },
                                                "first_3_values": {
                                                    "type": "array",
                                                    "items": {
                                                        "type": "string"
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            },
                            "description": "The following summary of the available data is returned:"
                        }
                    }
                }
            },
            {
                "description": "Return only the distinct elements from the input list.",
                "name": "select_unique_values",
                "parameters": {
                    "properties": {
                        "unique_array": {
                            "description": "A list of input data",
                            "schema": {
                                "type": "array"
                            }
                        }
                    },
                    "required": [
                        "unique_array"
                    ],
                    "type": "object"
                },
                "output_parameters": {
                    "properties": {
                        "output_0": {
                            "description": "The distinct elements of the input list",
                            "type": "array"
                        }
                    }
                }
            },
            {
                "description": "Transform list of string values by taking substrings",
                "name": "transform_data_to_substring",
                "parameters": {
                    "properties": {
                        "data_source": {
                            "description": "The location of the data file in csv format.",
                            "schema": {
                                "type": "string"
                            }
                        },
                        "key_name": {
                            "description": "name of string valued key to transform: must be chosen from the ALLOWED_VALUES_FOR_KEY_NAME enum. ",
                            "schema": {
                                "type": "string",
                                "enum": [
                                    "ALLOWED_VALUES_FOR_KEY_NAME"
                                ]
                            }
                        },
                        "start_index": {
                            "description": "start of substring",
                            "schema": {
                                "type": "integer"
                            }
                        },
                        "end_index": {
                            "description": "end of substring, must be >= start_index",
                            "schema": {
                                "type": "integer"
                            }
                        }
                    },
                    "required": [
                        "data_source",
                        "key_name",
                        "start_index",
                        "end_index"
                    ],
                    "type": "object"
                },
                "output_parameters": {
                    "properties": {
                        "output_0": {
                            "type": "object",
                            "properties": {
                                "data_path": {
                                    "type": "string",
                                    "description": "location where full data can be accessed."
                                },
                                "num_records": {
                                    "type": "string",
                                    "description": "number of individual records"
                                },
                                "keys": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "key_details": {
                                    "type": "object",
                                    "properties": {
                                        "key1": {
                                            "type": "object",
                                            "properties": {
                                                "dtype": {
                                                    "type": "string",
                                                    "description": "\"int64\""
                                                },
                                                "first_3_values": {
                                                    "type": "array",
                                                    "items": {
                                                        "type": "string"
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            },
                            "description": "The following summary of the available data is returned:"
                        }
                    }
                }
            },
            {
                "description": "Transform numeric values into their absolute value.",
                "name": "transform_data_to_absolute_value",
                "parameters": {
                    "properties": {
                        "data_source": {
                            "description": "The location of the data file in csv format.",
                            "schema": {
                                "type": "string"
                            }
                        },
                        "key_name": {
                            "description": "name of string valued key to take absolute value of: must be chosen from the ALLOWED_VALUES_FOR_KEY_NAME enum. ",
                            "schema": {
                                "type": "string",
                                "enum": [
                                    "ALLOWED_VALUES_FOR_KEY_NAME"
                                ]
                            }
                        }
                    },
                    "required": [
                        "data_source",
                        "key_name"
                    ],
                    "type": "object"
                },
                "output_parameters": {
                    "properties": {
                        "output_0": {
                            "type": "object",
                            "properties": {
                                "data_path": {
                                    "type": "string",
                                    "description": "location where full data can be accessed."
                                },
                                "num_records": {
                                    "type": "string",
                                    "description": "number of individual records"
                                },
                                "keys": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "key_details": {
                                    "type": "object",
                                    "properties": {
                                        "key1": {
                                            "type": "object",
                                            "properties": {
                                                "dtype": {
                                                    "type": "string",
                                                    "description": "\"int64\""
                                                },
                                                "first_3_values": {
                                                    "type": "array",
                                                    "items": {
                                                        "type": "string"
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            },
                            "description": "The following summary of the available data is returned:"
                        }
                    }
                }
            },
            {
                "description": "Transform list of datetime strings into specified subpart.",
                "name": "transform_data_to_datetime_part",
                "parameters": {
                    "properties": {
                        "data_source": {
                            "description": "The location of the data file in csv format.",
                            "schema": {
                                "type": "string"
                            }
                        },
                        "key_name": {
                            "description": "name of string valued key to transform: must be chosen from the ALLOWED_VALUES_FOR_KEY_NAME enum. ",
                            "schema": {
                                "type": "string",
                                "enum": [
                                    "ALLOWED_VALUES_FOR_KEY_NAME"
                                ]
                            }
                        },
                        "datetime_pattern": {
                            "description": "ISO datetime pattern to extract",
                            "schema": {
                                "type": "string"
                            }
                        }
                    },
                    "required": [
                        "data_source",
                        "key_name",
                        "datetime_pattern"
                    ],
                    "type": "object"
                },
                "output_parameters": {
                    "properties": {
                        "output_0": {
                            "type": "object",
                            "properties": {
                                "data_path": {
                                    "type": "string",
                                    "description": "location where full data can be accessed."
                                },
                                "num_records": {
                                    "type": "string",
                                    "description": "number of individual records"
                                },
                                "keys": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "key_details": {
                                    "type": "object",
                                    "properties": {
                                        "key1": {
                                            "type": "object",
                                            "properties": {
                                                "dtype": {
                                                    "type": "string",
                                                    "description": "\"int64\""
                                                },
                                                "first_3_values": {
                                                    "type": "array",
                                                    "items": {
                                                        "type": "string"
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            },
                            "description": "The following summary of the available data is returned:"
                        }
                    }
                }
            },
            {
                "description": "Return the first `n` elements of a list-like object.",
                "name": "truncate",
                "parameters": {
                    "properties": {
                        "truncate_array": {
                            "description": "A list-like object.",
                            "schema": {
                                "type": "object"
                            }
                        },
                        "n": {
                            "description": "The number of rows/elements to return.",
                            "schema": {
                                "type": "integer"
                            }
                        }
                    },
                    "required": [
                        "truncate_array",
                        "n"
                    ],
                    "type": "object"
                },
                "output_parameters": {
                    "properties": {
                        "output_0": {
                            "description": "The first `n` elements of the list-like object.",
                            "type": "array"
                        }
                    }
                }
            },
            {
                "description": "Filters rows where the column's value is equal to the given value.",
                "name": "select_data_equal_to",
                "parameters": {
                    "properties": {
                        "data_source": {
                            "description": "The location of the data file in csv format.",
                            "schema": {
                                "type": "string"
                            }
                        },
                        "key_name": {
                            "description": "The key on which the filter will be applied.: must be chosen from the ALLOWED_VALUES_FOR_KEY_NAME enum. ",
                            "schema": {
                                "type": "string",
                                "enum": [
                                    "ALLOWED_VALUES_FOR_KEY_NAME"
                                ]
                            }
                        },
                        "value": {
                            "description": "The value to compare against in the filtering operation.",
                            "schema": {
                                "type": "object"
                            }
                        }
                    },
                    "required": [
                        "data_source",
                        "key_name",
                        "value"
                    ],
                    "type": "object"
                },
                "output_parameters": {
                    "properties": {
                        "output_0": {
                            "type": "object",
                            "properties": {
                                "data_path": {
                                    "type": "string",
                                    "description": "location where full data can be accessed."
                                },
                                "num_records": {
                                    "type": "string",
                                    "description": "number of individual records"
                                },
                                "keys": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "key_details": {
                                    "type": "object",
                                    "properties": {
                                        "key1": {
                                            "type": "object",
                                            "properties": {
                                                "dtype": {
                                                    "type": "string",
                                                    "description": "\"int64\""
                                                },
                                                "first_3_values": {
                                                    "type": "array",
                                                    "items": {
                                                        "type": "string"
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            },
                            "description": "The following summary of the available data is returned:"
                        }
                    }
                }
            },
            {
                "description": "Filters rows where the column's value is not equal to the given value.",
                "name": "select_data_not_equal_to",
                "parameters": {
                    "properties": {
                        "data_source": {
                            "description": "The location of the data file in csv format.",
                            "schema": {
                                "type": "string"
                            }
                        },
                        "key_name": {
                            "description": "The key on which the filter will be applied.: must be chosen from the ALLOWED_VALUES_FOR_KEY_NAME enum. ",
                            "schema": {
                                "type": "string",
                                "enum": [
                                    "ALLOWED_VALUES_FOR_KEY_NAME"
                                ]
                            }
                        },
                        "value": {
                            "description": "The value to compare against in the filtering operation.",
                            "schema": {
                                "type": "object"
                            }
                        }
                    },
                    "required": [
                        "data_source",
                        "key_name",
                        "value"
                    ],
                    "type": "object"
                },
                "output_parameters": {
                    "properties": {
                        "output_0": {
                            "type": "object",
                            "properties": {
                                "data_path": {
                                    "type": "string",
                                    "description": "location where full data can be accessed."
                                },
                                "num_records": {
                                    "type": "string",
                                    "description": "number of individual records"
                                },
                                "keys": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "key_details": {
                                    "type": "object",
                                    "properties": {
                                        "key1": {
                                            "type": "object",
                                            "properties": {
                                                "dtype": {
                                                    "type": "string",
                                                    "description": "\"int64\""
                                                },
                                                "first_3_values": {
                                                    "type": "array",
                                                    "items": {
                                                        "type": "string"
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            },
                            "description": "The following summary of the available data is returned:"
                        }
                    }
                }
            },
            {
                "description": "Filters rows where the column's value is greater than the given value.",
                "name": "select_data_greater_than",
                "parameters": {
                    "properties": {
                        "data_source": {
                            "description": "The location of the data file in csv format.",
                            "schema": {
                                "type": "string"
                            }
                        },
                        "key_name": {
                            "description": "The key on which the filter will be applied.: must be chosen from the ALLOWED_VALUES_FOR_KEY_NAME enum. ",
                            "schema": {
                                "type": "string",
                                "enum": [
                                    "ALLOWED_VALUES_FOR_KEY_NAME"
                                ]
                            }
                        },
                        "value": {
                            "description": "The value to compare against in the filtering operation.",
                            "schema": {
                                "type": "object"
                            }
                        }
                    },
                    "required": [
                        "data_source",
                        "key_name",
                        "value"
                    ],
                    "type": "object"
                },
                "output_parameters": {
                    "properties": {
                        "output_0": {
                            "type": "object",
                            "properties": {
                                "data_path": {
                                    "type": "string",
                                    "description": "location where full data can be accessed."
                                },
                                "num_records": {
                                    "type": "string",
                                    "description": "number of individual records"
                                },
                                "keys": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "key_details": {
                                    "type": "object",
                                    "properties": {
                                        "key1": {
                                            "type": "object",
                                            "properties": {
                                                "dtype": {
                                                    "type": "string",
                                                    "description": "\"int64\""
                                                },
                                                "first_3_values": {
                                                    "type": "array",
                                                    "items": {
                                                        "type": "string"
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            },
                            "description": "The following summary of the available data is returned:"
                        }
                    }
                }
            },
            {
                "description": "Filters rows where the column's value is less than the given value.",
                "name": "select_data_less_than",
                "parameters": {
                    "properties": {
                        "data_source": {
                            "description": "The location of the data file in csv format.",
                            "schema": {
                                "type": "string"
                            }
                        },
                        "key_name": {
                            "description": "The key on which the filter will be applied.: must be chosen from the ALLOWED_VALUES_FOR_KEY_NAME enum. ",
                            "schema": {
                                "type": "string",
                                "enum": [
                                    "ALLOWED_VALUES_FOR_KEY_NAME"
                                ]
                            }
                        },
                        "value": {
                            "description": "The value to compare against in the filtering operation.",
                            "schema": {
                                "type": "object"
                            }
                        }
                    },
                    "required": [
                        "data_source",
                        "key_name",
                        "value"
                    ],
                    "type": "object"
                },
                "output_parameters": {
                    "properties": {
                        "output_0": {
                            "type": "object",
                            "properties": {
                                "data_path": {
                                    "type": "string",
                                    "description": "location where full data can be accessed."
                                },
                                "num_records": {
                                    "type": "string",
                                    "description": "number of individual records"
                                },
                                "keys": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "key_details": {
                                    "type": "object",
                                    "properties": {
                                        "key1": {
                                            "type": "object",
                                            "properties": {
                                                "dtype": {
                                                    "type": "string",
                                                    "description": "\"int64\""
                                                },
                                                "first_3_values": {
                                                    "type": "array",
                                                    "items": {
                                                        "type": "string"
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            },
                            "description": "The following summary of the available data is returned:"
                        }
                    }
                }
            },
            {
                "description": "Filters rows where the column's value is greater than or equal to the given value.",
                "name": "select_data_greater_than_equal_to",
                "parameters": {
                    "properties": {
                        "data_source": {
                            "description": "The location of the data file in csv format.",
                            "schema": {
                                "type": "string"
                            }
                        },
                        "key_name": {
                            "description": "The key on which the filter will be applied.: must be chosen from the ALLOWED_VALUES_FOR_KEY_NAME enum. ",
                            "schema": {
                                "type": "string",
                                "enum": [
                                    "ALLOWED_VALUES_FOR_KEY_NAME"
                                ]
                            }
                        },
                        "value": {
                            "description": "The value to compare against in the filtering operation.",
                            "schema": {
                                "type": "object"
                            }
                        }
                    },
                    "required": [
                        "data_source",
                        "key_name",
                        "value"
                    ],
                    "type": "object"
                },
                "output_parameters": {
                    "properties": {
                        "output_0": {
                            "type": "object",
                            "properties": {
                                "data_path": {
                                    "type": "string",
                                    "description": "location where full data can be accessed."
                                },
                                "num_records": {
                                    "type": "string",
                                    "description": "number of individual records"
                                },
                                "keys": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "key_details": {
                                    "type": "object",
                                    "properties": {
                                        "key1": {
                                            "type": "object",
                                            "properties": {
                                                "dtype": {
                                                    "type": "string",
                                                    "description": "\"int64\""
                                                },
                                                "first_3_values": {
                                                    "type": "array",
                                                    "items": {
                                                        "type": "string"
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            },
                            "description": "The following summary of the available data is returned:"
                        }
                    }
                }
            },
            {
                "description": "Filters rows where the column's value is less than or equal to the given value.",
                "name": "select_data_less_than_equal_to",
                "parameters": {
                    "properties": {
                        "data_source": {
                            "description": "The location of the data file in csv format.",
                            "schema": {
                                "type": "string"
                            }
                        },
                        "key_name": {
                            "description": "The key on which the filter will be applied.: must be chosen from the ALLOWED_VALUES_FOR_KEY_NAME enum. ",
                            "schema": {
                                "type": "string",
                                "enum": [
                                    "ALLOWED_VALUES_FOR_KEY_NAME"
                                ]
                            }
                        },
                        "value": {
                            "description": "The value to compare against in the filtering operation.",
                            "schema": {
                                "type": "object"
                            }
                        }
                    },
                    "required": [
                        "data_source",
                        "key_name",
                        "value"
                    ],
                    "type": "object"
                },
                "output_parameters": {
                    "properties": {
                        "output_0": {
                            "type": "object",
                            "properties": {
                                "data_path": {
                                    "type": "string",
                                    "description": "location where full data can be accessed."
                                },
                                "num_records": {
                                    "type": "string",
                                    "description": "number of individual records"
                                },
                                "keys": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "key_details": {
                                    "type": "object",
                                    "properties": {
                                        "key1": {
                                            "type": "object",
                                            "properties": {
                                                "dtype": {
                                                    "type": "string",
                                                    "description": "\"int64\""
                                                },
                                                "first_3_values": {
                                                    "type": "array",
                                                    "items": {
                                                        "type": "string"
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            },
                            "description": "The following summary of the available data is returned:"
                        }
                    }
                }
            },
            {
                "description": "Filters rows where the column's value contains the given value (applies to strings).",
                "name": "select_data_contains",
                "parameters": {
                    "properties": {
                        "data_source": {
                            "description": "The location of the data file in csv format.",
                            "schema": {
                                "type": "string"
                            }
                        },
                        "key_name": {
                            "description": "The key on which the filter will be applied.: must be chosen from the ALLOWED_VALUES_FOR_KEY_NAME enum. ",
                            "schema": {
                                "type": "string",
                                "enum": [
                                    "ALLOWED_VALUES_FOR_KEY_NAME"
                                ]
                            }
                        },
                        "value": {
                            "description": "The value to compare against in the filtering operation.",
                            "schema": {
                                "type": "object"
                            }
                        }
                    },
                    "required": [
                        "data_source",
                        "key_name",
                        "value"
                    ],
                    "type": "object"
                },
                "output_parameters": {
                    "properties": {
                        "output_0": {
                            "type": "object",
                            "properties": {
                                "data_path": {
                                    "type": "string",
                                    "description": "location where full data can be accessed."
                                },
                                "num_records": {
                                    "type": "string",
                                    "description": "number of individual records"
                                },
                                "keys": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "key_details": {
                                    "type": "object",
                                    "properties": {
                                        "key1": {
                                            "type": "object",
                                            "properties": {
                                                "dtype": {
                                                    "type": "string",
                                                    "description": "\"int64\""
                                                },
                                                "first_3_values": {
                                                    "type": "array",
                                                    "items": {
                                                        "type": "string"
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            },
                            "description": "The following summary of the available data is returned:"
                        }
                    }
                }
            },
            {
                "description": "Filters rows where the column's value matches a regex pattern (applies to strings).",
                "name": "select_data_like",
                "parameters": {
                    "properties": {
                        "data_source": {
                            "description": "The location of the data file in csv format.",
                            "schema": {
                                "type": "string"
                            }
                        },
                        "key_name": {
                            "description": "The key on which the filter will be applied.: must be chosen from the ALLOWED_VALUES_FOR_KEY_NAME enum. ",
                            "schema": {
                                "type": "string",
                                "enum": [
                                    "ALLOWED_VALUES_FOR_KEY_NAME"
                                ]
                            }
                        },
                        "value": {
                            "description": "The value to compare against in the filtering operation.",
                            "schema": {
                                "type": "object"
                            }
                        }
                    },
                    "required": [
                        "data_source",
                        "key_name",
                        "value"
                    ],
                    "type": "object"
                },
                "output_parameters": {
                    "properties": {
                        "output_0": {
                            "type": "object",
                            "properties": {
                                "data_path": {
                                    "type": "string",
                                    "description": "location where full data can be accessed."
                                },
                                "num_records": {
                                    "type": "string",
                                    "description": "number of individual records"
                                },
                                "keys": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "key_details": {
                                    "type": "object",
                                    "properties": {
                                        "key1": {
                                            "type": "object",
                                            "properties": {
                                                "dtype": {
                                                    "type": "string",
                                                    "description": "\"int64\""
                                                },
                                                "first_3_values": {
                                                    "type": "array",
                                                    "items": {
                                                        "type": "string"
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            },
                            "description": "The following summary of the available data is returned:"
                        }
                    }
                }
            },
            {
                "description": "Sort data in ascending order by the values associated with the chosen key='key_name' If the input data is list-like, returns the sorted list. If the input data is tabular, returns the table with rows sorted by the values in column 'key_name'. If the data is grouped tables, then sort the groups by the value in 'key_name'",
                "name": "sort_data_ascending",
                "parameters": {
                    "properties": {
                        "data_source": {
                            "description": "The location of the data file in csv format.",
                            "schema": {
                                "type": "string"
                            }
                        },
                        "key_name": {
                            "description": "name of key to sort by: must be chosen from the ALLOWED_VALUES_FOR_KEY_NAME enum. ",
                            "schema": {
                                "type": "string",
                                "enum": [
                                    "ALLOWED_VALUES_FOR_KEY_NAME"
                                ]
                            }
                        }
                    },
                    "required": [
                        "data_source",
                        "key_name"
                    ],
                    "type": "object"
                },
                "output_parameters": {
                    "properties": {
                        "output_0": {
                            "type": "object",
                            "properties": {
                                "data_path": {
                                    "type": "string",
                                    "description": "location where full data can be accessed."
                                },
                                "num_records": {
                                    "type": "string",
                                    "description": "number of individual records"
                                },
                                "keys": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "key_details": {
                                    "type": "object",
                                    "properties": {
                                        "key1": {
                                            "type": "object",
                                            "properties": {
                                                "dtype": {
                                                    "type": "string",
                                                    "description": "\"int64\""
                                                },
                                                "first_3_values": {
                                                    "type": "array",
                                                    "items": {
                                                        "type": "string"
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            },
                            "description": "The following summary of the available data is returned:"
                        }
                    }
                }
            },
            {
                "description": "Sort data in descending order by the values associated with the chosen key='key_name' If the input data is list-like, returns the sorted list. If the input data is tabular, returns the table with rows sorted by the values in column 'key_name'. If the data is grouped tables, then sort the groups by the value in 'key_name'",
                "name": "sort_data_descending",
                "parameters": {
                    "properties": {
                        "data_source": {
                            "description": "The location of the data file in csv format.",
                            "schema": {
                                "type": "string"
                            }
                        },
                        "key_name": {
                            "description": "name of key to sort by: must be chosen from the ALLOWED_VALUES_FOR_KEY_NAME enum. ",
                            "schema": {
                                "type": "string",
                                "enum": [
                                    "ALLOWED_VALUES_FOR_KEY_NAME"
                                ]
                            }
                        }
                    },
                    "required": [
                        "data_source",
                        "key_name"
                    ],
                    "type": "object"
                },
                "output_parameters": {
                    "properties": {
                        "output_0": {
                            "type": "object",
                            "properties": {
                                "data_path": {
                                    "type": "string",
                                    "description": "location where full data can be accessed."
                                },
                                "num_records": {
                                    "type": "string",
                                    "description": "number of individual records"
                                },
                                "keys": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "key_details": {
                                    "type": "object",
                                    "properties": {
                                        "key1": {
                                            "type": "object",
                                            "properties": {
                                                "dtype": {
                                                    "type": "string",
                                                    "description": "\"int64\""
                                                },
                                                "first_3_values": {
                                                    "type": "array",
                                                    "items": {
                                                        "type": "string"
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            },
                            "description": "The following summary of the available data is returned:"
                        }
                    }
                }
            },
            {
                "description": "Return the min of the input data over the specified key.",
                "name": "compute_data_min",
                "parameters": {
                    "properties": {
                        "data_source": {
                            "description": "The location of the data file in csv format.",
                            "schema": {
                                "type": "string"
                            }
                        },
                        "key_name": {
                            "description": "name of key to apply min: must be chosen from the ALLOWED_VALUES_FOR_KEY_NAME enum. ",
                            "schema": {
                                "type": "string",
                                "enum": [
                                    "ALLOWED_VALUES_FOR_KEY_NAME"
                                ]
                            }
                        },
                        "distinct": {
                            "description": "whether to apply min to only distinct values",
                            "schema": {
                                "type": "boolean"
                            }
                        }
                    },
                    "required": [
                        "data_source",
                        "key_name",
                        "distinct"
                    ],
                    "type": "object"
                },
                "output_parameters": {
                    "properties": {
                        "output_0": {
                            "description": "Result of min."
                        }
                    }
                }
            },
            {
                "description": "Return the max of the input data over the specified key.",
                "name": "compute_data_max",
                "parameters": {
                    "properties": {
                        "data_source": {
                            "description": "The location of the data file in csv format.",
                            "schema": {
                                "type": "string"
                            }
                        },
                        "key_name": {
                            "description": "name of key to apply max: must be chosen from the ALLOWED_VALUES_FOR_KEY_NAME enum. ",
                            "schema": {
                                "type": "string",
                                "enum": [
                                    "ALLOWED_VALUES_FOR_KEY_NAME"
                                ]
                            }
                        },
                        "distinct": {
                            "description": "whether to apply max to only distinct values",
                            "schema": {
                                "type": "boolean"
                            }
                        }
                    },
                    "required": [
                        "data_source",
                        "key_name",
                        "distinct"
                    ],
                    "type": "object"
                },
                "output_parameters": {
                    "properties": {
                        "output_0": {
                            "description": "Result of max."
                        }
                    }
                }
            },
            {
                "description": "Return the sum of the input data over the specified key.",
                "name": "compute_data_sum",
                "parameters": {
                    "properties": {
                        "data_source": {
                            "description": "The location of the data file in csv format.",
                            "schema": {
                                "type": "string"
                            }
                        },
                        "key_name": {
                            "description": "name of key to apply sum: must be chosen from the ALLOWED_VALUES_FOR_KEY_NAME enum. ",
                            "schema": {
                                "type": "string",
                                "enum": [
                                    "ALLOWED_VALUES_FOR_KEY_NAME"
                                ]
                            }
                        },
                        "distinct": {
                            "description": "whether to apply sum to only distinct values",
                            "schema": {
                                "type": "boolean"
                            }
                        }
                    },
                    "required": [
                        "data_source",
                        "key_name",
                        "distinct"
                    ],
                    "type": "object"
                },
                "output_parameters": {
                    "properties": {
                        "output_0": {
                            "description": "Result of sum."
                        }
                    }
                }
            },
            {
                "description": "Return the mean of the input data over the specified key.",
                "name": "compute_data_mean",
                "parameters": {
                    "properties": {
                        "data_source": {
                            "description": "The location of the data file in csv format.",
                            "schema": {
                                "type": "string"
                            }
                        },
                        "key_name": {
                            "description": "name of key to apply mean: must be chosen from the ALLOWED_VALUES_FOR_KEY_NAME enum. ",
                            "schema": {
                                "type": "string",
                                "enum": [
                                    "ALLOWED_VALUES_FOR_KEY_NAME"
                                ]
                            }
                        },
                        "distinct": {
                            "description": "whether to apply mean to only distinct values",
                            "schema": {
                                "type": "boolean"
                            }
                        }
                    },
                    "required": [
                        "data_source",
                        "key_name",
                        "distinct"
                    ],
                    "type": "object"
                },
                "output_parameters": {
                    "properties": {
                        "output_0": {
                            "description": "Result of mean."
                        }
                    }
                }
            },
            {
                "description": "Return the count of the input data over the specified key.",
                "name": "compute_data_count",
                "parameters": {
                    "properties": {
                        "data_source": {
                            "description": "The location of the data file in csv format.",
                            "schema": {
                                "type": "string"
                            }
                        },
                        "key_name": {
                            "description": "name of key to apply count: must be chosen from the ALLOWED_VALUES_FOR_KEY_NAME enum. ",
                            "schema": {
                                "type": "string",
                                "enum": [
                                    "ALLOWED_VALUES_FOR_KEY_NAME"
                                ]
                            }
                        },
                        "distinct": {
                            "description": "whether to apply count to only distinct values",
                            "schema": {
                                "type": "boolean"
                            }
                        }
                    },
                    "required": [
                        "data_source",
                        "key_name",
                        "distinct"
                    ],
                    "type": "object"
                },
                "output_parameters": {
                    "properties": {
                        "output_0": {
                            "description": "Result of count."
                        }
                    }
                }
            },
            {
                "description": "Return the std of the input data over the specified key.",
                "name": "compute_data_std",
                "parameters": {
                    "properties": {
                        "data_source": {
                            "description": "The location of the data file in csv format.",
                            "schema": {
                                "type": "string"
                            }
                        },
                        "key_name": {
                            "description": "name of key to apply std: must be chosen from the ALLOWED_VALUES_FOR_KEY_NAME enum. ",
                            "schema": {
                                "type": "string",
                                "enum": [
                                    "ALLOWED_VALUES_FOR_KEY_NAME"
                                ]
                            }
                        },
                        "distinct": {
                            "description": "whether to apply std to only distinct values",
                            "schema": {
                                "type": "boolean"
                            }
                        }
                    },
                    "required": [
                        "data_source",
                        "key_name",
                        "distinct"
                    ],
                    "type": "object"
                },
                "output_parameters": {
                    "properties": {
                        "output_0": {
                            "description": "Result of std."
                        }
                    }
                }
            },
            {
                "description": "Return the argmin of the input data over the specified key.",
                "name": "compute_data_argmin",
                "parameters": {
                    "properties": {
                        "data_source": {
                            "description": "The location of the data file in csv format.",
                            "schema": {
                                "type": "string"
                            }
                        },
                        "key_name": {
                            "description": "name of key to apply argmin: must be chosen from the ALLOWED_VALUES_FOR_KEY_NAME enum. ",
                            "schema": {
                                "type": "string",
                                "enum": [
                                    "ALLOWED_VALUES_FOR_KEY_NAME"
                                ]
                            }
                        },
                        "distinct": {
                            "description": "whether to apply argmin to only distinct values",
                            "schema": {
                                "type": "boolean"
                            }
                        }
                    },
                    "required": [
                        "data_source",
                        "key_name",
                        "distinct"
                    ],
                    "type": "object"
                },
                "output_parameters": {
                    "properties": {
                        "output_0": {
                            "description": "Result of argmin."
                        }
                    }
                }
            },
            {
                "description": "Return the argmax of the input data over the specified key.",
                "name": "compute_data_argmax",
                "parameters": {
                    "properties": {
                        "data_source": {
                            "description": "The location of the data file in csv format.",
                            "schema": {
                                "type": "string"
                            }
                        },
                        "key_name": {
                            "description": "name of key to apply argmax: must be chosen from the ALLOWED_VALUES_FOR_KEY_NAME enum. ",
                            "schema": {
                                "type": "string",
                                "enum": [
                                    "ALLOWED_VALUES_FOR_KEY_NAME"
                                ]
                            }
                        },
                        "distinct": {
                            "description": "whether to apply argmax to only distinct values",
                            "schema": {
                                "type": "boolean"
                            }
                        }
                    },
                    "required": [
                        "data_source",
                        "key_name",
                        "distinct"
                    ],
                    "type": "object"
                },
                "output_parameters": {
                    "properties": {
                        "output_0": {
                            "description": "Result of argmax."
                        }
                    }
                }
            }
        ]
    }
]