import os
from typing import AnyStr

import mongoengine


def mongo_global_init(host: AnyStr, port: int = 27017, host_type: AnyStr = "local"):
    """
    Initialises mongo db connection which is available globally for
    every document object, syntax for defining a DOM (document object mapper):
    Extend the class from mongoengine.Document and initialise below dict
    in a variable named meta     = {
        "db_alias": 'core',
        "collection": 'candidate_ratings'
    }
    :param host: Host of the mongodb cluster
    :param port: Port of the cluster
    :param host_type: to use mongo cluster use - "cloud",
    to use the mongo local instance use - "local"
    default port for both is 27017
    :return: None
    """
    if host_type == "local":
        alias_core = "core"
        db_name = "sa_app_db"
        data = {"host": "localhost", "port": 27017}
        mongoengine.register_connection(alias=alias_core, name=db_name, **data)
    else:
        print(f"Host type is {host_type}")
        alias_core = "core"
        db_name = "sa_app_db"
        username = os.environ["mongo_user"]
        password = os.environ["mongo_pass"]
        data = {
            "host": f"mongodb+srv://{username}:{password}@{host}",
            "port": int(port),
            "username": username,
            "password": password,
        }
        mongoengine.register_connection(alias=alias_core, name=db_name, **data)
