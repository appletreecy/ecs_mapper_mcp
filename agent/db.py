import os
import mysql.connector
from dotenv import load_dotenv

load_dotenv()

def get_conn():
    return mysql.connector.connect(
        host=os.getenv("MYSQL_HOST", "127.0.0.1"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
        user=os.getenv("MYSQL_USER", "ecs_user"),
        password=os.getenv("MYSQL_PASSWORD", "ecspass"),
        database=os.getenv("MYSQL_DB", "ecs_mapper"),
        autocommit=True,
    )
