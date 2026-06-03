import json
import os
from threading import Lock

DATA_FILE = "data/cases.json"
lock = Lock()


def ensure_data_file():
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(DATA_FILE):
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump({}, f, indent=2)


def read_cases():
    ensure_data_file()
    with lock:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                return {}
            return json.loads(content)


def write_cases(data):
    ensure_data_file()
    with lock:
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


def save_case(case_id, payload):
    data = read_cases()
    data[case_id] = payload
    write_cases(data)
    return data[case_id]


def get_case(case_id):
    data = read_cases()
    return data.get(case_id)


def get_all_cases():
    return read_cases()