from app.main import app
from app.database import create_db_entry, print_db_entry, delete_db_entry
from fastapi import FastAPI
from pydantic import BaseModel


@app.post("/entry")
def database_entry(entry_id: int, entry: str):
    create_db_entry(entry_id,entry)
    return {"status":"successful entry made","entry_id":entry_id,"entry":entry}

entry_id = int(input("Enter ID: "))
entry = input("Enter text: ")

create_db_entry(entry_id,entry)
print_db_entry()
delete_db_entry(entry_id)
create_db_entry(entry_id,entry)
print_db_entry()