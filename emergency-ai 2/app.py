from flask import Flask, request, jsonify, render_template
from datetime import datetime, timedelta
from flask import send_file
from openai import OpenAI
from io import BytesIO
import json
import os
import re
import hashlib
import subprocess
import uuid
import threading
import time
import requests
from requests import HTTPError

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FLASK_BASE_URL = os.getenv("FLASK_BASE_URL", "http://127.0.0.1:5001").rstrip("/")
N8N_BASE_URL = os.getenv("N8N_BASE_URL", "http://localhost:5678/webhook").rstrip("/")

from storage import save_case, get_case, get_all_cases

app = Flask(__name__, static_folder="static", template_folder="templates")


def now_iso():
    return datetime.utcnow().isoformat()


SIMULATED_TIMELINE_OFFSETS = {
    "clara_at": 2,
    "ares_at": 6,
    "vita_pre_at": 14,
    "hospital_at": 38,
    "athena_at": 52,
    "vita_final_at": 67,
}

CASE_SCHEMA_VERSION = 2

STATUS_LABELS = {
    "idle": "in attesa",
    "running": "in elaborazione",
    "completed": "completato",
    "error": "errore",
}

MANUAL_PRIORITY_STYLES = {
    "ROSSO": "red",
    "GIALLO": "yellow",
    "VERDE": "green",
    "BIANCO": "white",
}

TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
TTS_VOICE = os.getenv("OPENAI_TTS_VOICE", "alloy")
TTS_TIMEOUT_SECONDS = float(os.getenv("OPENAI_TTS_TIMEOUT_SECONDS", "45"))
N8N_REQUEST_TIMEOUT_SECONDS = float(os.getenv("N8N_REQUEST_TIMEOUT_SECONDS", "120"))
DEFAULT_RHUBARB_BINARY = os.path.join(
    BASE_DIR,
    "tools",
    "rhubarb",
    "Rhubarb-Lip-Sync-1.14.0-macOS",
    "rhubarb",
)
RHUBARB_BINARY = os.getenv(
    "RHUBARB_BINARY",
    DEFAULT_RHUBARB_BINARY if os.path.exists(DEFAULT_RHUBARB_BINARY) else "rhubarb",
)
AVATAR_AUDIO_DIR = os.path.join(BASE_DIR, "static", "generated", "avatar_audio")


def get_simulated_timeline_timestamp(case, timeline_key):
    base_timestamp = (
        case.get("timeline", {}).get("created_at")
        or case.get("created_at")
        or now_iso()
    )

    try:
        base_dt = datetime.fromisoformat(base_timestamp)
    except ValueError:
        base_dt = datetime.utcnow()

    offset_minutes = SIMULATED_TIMELINE_OFFSETS.get(timeline_key, 0)
    return (base_dt + timedelta(minutes=offset_minutes)).isoformat()


def normalize_tts_text(text):
    normalized = text or ""
    replacements = [
        (r"\*\*(.*?)\*\*", r"\1"),
        (r"\*(.*?)\*", r"\1"),
        (r"`([^`]+)`", r"\1"),
        (r"\bSpO2\b", "saturazione di ossigeno"),
        (r"\bSaO2\b", "saturazione arteriosa di ossigeno"),
        (r"\bmmHg\b", " millimetri di mercurio"),
        (r"\bbpm\b", " battiti per minuto"),
        (r"\bECG\b", "elettrocardiogramma"),
        (r"\bCBC\b", "emocromo"),
        (r"\bSTEMI\b", "stemi, infarto miocardico con sopraslivellamento del tratto esse ti"),
        (r"\bNSTE-?ACS\b", "sindrome coronarica acuta senza sopraslivellamento del tratto esse ti"),
        (r"\bICU\b", "terapia intensiva"),
        (r"\bPS\b", "pronto soccorso"),
        (r"\bPA\b", "pressione arteriosa"),
        (r"\bHR\b", "frequenza cardiaca"),
        (r"\bO2\b", "ossigeno"),
        (r"\bTAC\b", "tac"),
        (r"\bRMN\b", "risonanza magnetica"),
        (r"\bIV\b", "endovena"),
        (r"\bIM\b", "intramuscolo"),
        (r"\bPO\b", "per os"),
        (r"\bN\/D\b", "non disponibile"),
    ]

    for pattern, replacement in replacements:
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)

    normalized = re.sub(
        r"\b([0-9]{2,3})\/([0-9]{2,3})\s+millimetri di mercurio\b",
        r"\1 su \2 millimetri di mercurio",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(r"\b([0-9]{2,3})\/([0-9]{2,3})\b", r"\1 su \2", normalized)
    normalized = re.sub(r"\b([0-9]+(?:[.,][0-9]+)?)%\b", r"\1 per cento", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\b([0-9]+(?:[.,][0-9]+)?)\s?°C\b", r"\1 gradi", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\b([0-9]+(?:[.,][0-9]+)?)\s?mg\/dl\b", r"\1 milligrammi per decilitro", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\b([0-9]+(?:[.,][0-9]+)?)\s?g\/dl\b", r"\1 grammi per decilitro", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\b([0-9]+(?:[.,][0-9]+)?)\s?mmol\/l\b", r"\1 millimoli per litro", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\b([0-9]+(?:[.,][0-9]+)?)\s?l\/min\b", r"\1 litri al minuto", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\b([0-9]+(?:[.,][0-9]+)?)\s?kg\b", r"\1 chilogrammi", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\b([0-9]+(?:[.,][0-9]+)?)\s?cm\b", r"\1 centimetri", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\n{2,}", ". ", normalized)
    normalized = re.sub(r"\n", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"\s([,.!?;:])", r"\1", normalized)
    return normalized.strip()


def ensure_avatar_audio_dir():
    os.makedirs(AVATAR_AUDIO_DIR, exist_ok=True)


def build_sentence_ranges(text):
    ranges = []
    for match in re.finditer(r"[^.!?]+[.!?]?", text or ""):
        sentence = match.group(0)
        if not sentence.strip():
            continue
        ranges.append({
            "start": match.start(),
            "end": match.end(),
            "text": sentence.strip(),
        })
    if not ranges and (text or "").strip():
        ranges.append({
            "start": 0,
            "end": len(text.strip()),
            "text": text.strip(),
        })
    return ranges


def build_fallback_rhubarb_cues(text, duration_seconds):
    sentence_ranges = build_sentence_ranges(text)
    if not sentence_ranges or not duration_seconds:
        return []

    total_chars = max(len(text or ""), 1)
    elapsed_seconds = 0.0
    cue_values = ["A", "B", "C", "D", "E", "F", "G"]
    cues = []

    for index, sentence in enumerate(sentence_ranges):
        portion = len(sentence["text"]) / total_chars
        start = elapsed_seconds
        end = elapsed_seconds + (duration_seconds * portion)
        elapsed_seconds = end
        cues.append({
            "start": round(start, 3),
            "end": round(end, 3),
            "value": cue_values[index % len(cue_values)],
        })

    return cues


def parse_rhubarb_json(json_path):
    with open(json_path, "r", encoding="utf-8") as file:
        payload = json.load(file)

    mouth_cues = payload.get("mouthCues", [])
    cues = []
    for index, cue in enumerate(mouth_cues):
        start = float(cue.get("start", 0))
        end = float(cue.get("end", start))
        if end <= start and index + 1 < len(mouth_cues):
            end = float(mouth_cues[index + 1].get("start", start))
        cues.append({
            "start": round(start, 3),
            "end": round(end, 3),
            "value": str(cue.get("value", "X")),
        })
    return cues


def generate_rhubarb_cues(audio_path, text, duration_seconds):
    json_path = f"{os.path.splitext(audio_path)[0]}.json"
    try:
        subprocess.run(
            [
                RHUBARB_BINARY,
                "-f", "json",
                "-o", json_path,
                audio_path,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        return {
            "provider": "rhubarb",
            "cues": parse_rhubarb_json(json_path),
        }
    except (FileNotFoundError, subprocess.SubprocessError, json.JSONDecodeError):
        return {
            "provider": "fallback",
            "cues": build_fallback_rhubarb_cues(text, duration_seconds),
        }


def estimate_audio_duration_seconds(audio_bytes):
    if len(audio_bytes) < 44:
        return 0.0

    if audio_bytes[:4] != b"RIFF" or audio_bytes[8:12] != b"WAVE":
        return 0.0

    offset = 12
    byte_rate = 0
    data_size = 0

    while offset + 8 <= len(audio_bytes):
        chunk_id = audio_bytes[offset:offset + 4]
        chunk_size = int.from_bytes(audio_bytes[offset + 4:offset + 8], "little")
        chunk_data_start = offset + 8
        chunk_data_end = chunk_data_start + chunk_size

        if chunk_id == b"fmt " and chunk_size >= 16:
            byte_rate = int.from_bytes(audio_bytes[chunk_data_start + 8:chunk_data_start + 12], "little")
        elif chunk_id == b"data":
            data_size = chunk_size
            break

        offset = chunk_data_end + (chunk_size % 2)

    if not byte_rate:
        return 0.0

    return round(data_size / byte_rate, 3)


def build_agent_input(case_data, agent_name):
    if agent_name == "clara":
        return {
            "case_id": case_data["case_id"],
            "phase": case_data["phase"],
            "patient_input": case_data["raw_input"]
        }

    if agent_name == "ares":
        return {
            "case_id": case_data["case_id"],
            "clara_output": case_data["clara_output"]
        }

    if agent_name == "vita-pre":
        return {
            "case_id": case_data["case_id"],
            "clara_output": case_data["clara_output"],
            "ares_output": case_data["ares_output"]
        }

    if agent_name == "athena":
        return {
            "case_id": case_data["case_id"],
            "clara_output": case_data["clara_output"],
            "ares_output": case_data["ares_output"],
            "hospital_input": case_data["hospital_input"]
        }

    if agent_name == "vita-final":
        return {
            "case_id": case_data["case_id"],
            "clara_output": case_data["clara_output"],
            "ares_output": case_data["ares_output"],
            "athena_output": case_data["athena_output"],
            "hospital_input": case_data["hospital_input"]
        }

    return None

def update_case(case_id, updater):
    case = get_case(case_id)
    if not case:
        return None
    updater(case)
    case["updated_at"] = now_iso()
    save_case(case_id, case)
    return case


def set_simulation_status(case_id, status):
    def updater(case):
        case["simulation_status"] = status
    update_case(case_id, updater)


def get_status_label(status):
    return STATUS_LABELS.get(status, status or "-")


def normalize_priority_code(priority_code):
    raw_value = (priority_code or "").strip().lower()
    return {
        "rosso": "ROSSO",
        "emergente": "ROSSO",
        "giallo": "GIALLO",
        "urgente": "GIALLO",
        "verde": "VERDE",
        "stabile": "VERDE",
        "bianco": "BIANCO",
        "non urgente": "BIANCO",
        "non-urgente": "BIANCO",
        "minore": "BIANCO",
        "banale": "BIANCO",
    }.get(raw_value, (priority_code or "").strip())


def normalize_demo_agent_payload(case, field_name, payload):
    normalized_payload = dict(payload or {})

    if case.get("demo_key") == "demo_bianco":
        if field_name == "ares_output":
            normalized_payload["priority_code"] = "bianco"

        if field_name == "vita_pre_output":
            for key in ("dashboard_summary", "operative_explanation", "clinical_comment"):
                value = normalized_payload.get(key)
                if not isinstance(value, str):
                    continue

                value = re.sub(r"\bpriorit[aà]\s+verde\b", "priorità bianco", value, flags=re.IGNORECASE)
                value = re.sub(r"\bcodice\s+priorit[aà]\s+verde\b", "codice priorità bianco", value, flags=re.IGNORECASE)
                value = re.sub(r"\bcodice\s+verde\b", "codice bianco", value, flags=re.IGNORECASE)
                normalized_payload[key] = value

    if case.get("demo_key") == "demo_verde":
        if field_name == "ares_output":
            normalized_payload["priority_code"] = "verde"

        if field_name == "vita_pre_output":
            for key in ("dashboard_summary", "operative_explanation", "clinical_comment"):
                value = normalized_payload.get(key)
                if not isinstance(value, str):
                    continue

                value = re.sub(r"\bpriorit[aà]\s+giall[oa]\b", "priorità verde", value, flags=re.IGNORECASE)
                value = re.sub(r"\bcodice\s+priorit[aà]\s+giall[oa]\b", "codice priorità verde", value, flags=re.IGNORECASE)
                value = re.sub(r"\bcodice\s+giall[oa]\b", "codice verde", value, flags=re.IGNORECASE)
                normalized_payload[key] = value

    return normalized_payload


def reset_case_for_restart(case_id):
    def updater(case):
        case["phase"] = "pre_hospital"
        case["current_step"] = "clara_pending"
        case["clara_output"] = None
        case["ares_output"] = None
        case["vita_pre_output"] = None
        case["hospital_input"] = None
        case["athena_output"] = None
        case["vita_final_output"] = None
        case["simulation_status"] = "idle"
        case["process_started"] = False
        case["manual_priority"] = None
        case["schema_version"] = CASE_SCHEMA_VERSION
        case.pop("simulation_error", None)

        timeline = case.setdefault("timeline", {})
        timeline["clara_at"] = None
        timeline["ares_at"] = None
        timeline["vita_pre_at"] = None
        timeline["hospital_at"] = None
        timeline["athena_at"] = None
        timeline["vita_final_at"] = None

    return update_case(case_id, updater)


def get_n8n_url(env_var_name, default_path):
    override = os.getenv(env_var_name)
    if override:
        return override.rstrip("/")
    return f"{N8N_BASE_URL}/{default_path}"


def post_json(url, payload, allow_read_timeout_as_accepted=False):
    try:
        response = requests.post(url, json=payload, timeout=N8N_REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
    except HTTPError as exc:
        response = exc.response
        status_code = response.status_code if response is not None else "unknown"
        response_text = ""
        if response is not None:
            response_text = (response.text or "").strip()
            response_text = response_text[:300]
        raise Exception(
            f"HTTP POST {url} failed with status {status_code}. Response: {response_text or 'empty body'}"
        ) from exc
    except requests.ReadTimeout:
        if allow_read_timeout_as_accepted:
            return {
                "status": "accepted_timeout",
                "message": (
                    f"HTTP POST {url} ha superato il timeout di lettura, "
                    "ma il workflow potrebbe essere gia stato preso in carico da n8n."
                ),
            }
        raise Exception(
            f"HTTP POST {url} failed: read timeout after {N8N_REQUEST_TIMEOUT_SECONDS} seconds"
        )
    except requests.RequestException as exc:
        raise Exception(f"HTTP POST {url} failed: {exc}") from exc

    if not response.content:
        return {"status": "success", "http_status": response.status_code}

    try:
        return response.json()
    except ValueError:
        return {
            "status": "success",
            "http_status": response.status_code,
            "raw_response": response.text
        }


def trigger_clara(case_id):
    return post_json(
        get_n8n_url("N8N_CLARA_URL", "clara-start"),
        {"case_id": case_id},
        allow_read_timeout_as_accepted=True,
    )


def trigger_ares(case_id):
    return post_json(
        get_n8n_url("N8N_ARES_URL", "ares-start"),
        {"case_id": case_id},
        allow_read_timeout_as_accepted=True,
    )


def trigger_vita_pre(case_id):
    return post_json(
        get_n8n_url("N8N_VITA_PRE_URL", "vita-pre-start"),
        {"case_id": case_id},
        allow_read_timeout_as_accepted=True,
    )


def trigger_athena(case_id):
    return post_json(
        get_n8n_url("N8N_ATHENA_URL", "athena-start"),
        {"case_id": case_id},
        allow_read_timeout_as_accepted=True,
    )


def trigger_vita_final(case_id):
    return post_json(
        get_n8n_url("N8N_VITA_FINAL_URL", "vita-final-start"),
        {"case_id": case_id},
        allow_read_timeout_as_accepted=True,
    )


def send_hospital_data(case_id, hospital_input):
    return post_json(f"{FLASK_BASE_URL}/api/cases/{case_id}/hospital-data", hospital_input)


def wait_for_case_fields(case_id, required_fields, timeout_seconds=30, poll_interval=1):
    deadline = time.time() + timeout_seconds

    while time.time() < deadline:
        case = get_case(case_id)
        if not case:
            return None

        if all(case.get(field) for field in required_fields):
            return case

        time.sleep(poll_interval)

    return get_case(case_id)


def run_case_simulation(case_id):
    try:
        case = get_case(case_id)
        if not case:
            return

        trigger_clara(case_id)
        time.sleep(2)

        trigger_ares(case_id)
        time.sleep(2)

        trigger_vita_pre(case_id)
        time.sleep(2)

        case = get_case(case_id)
        if not case:
            return

        if case.get("path_type") == "pre_hospital":
            case = wait_for_case_fields(case_id, ["clara_output", "ares_output", "vita_pre_output"])
            if not case:
                return

            if case.get("clara_output") and case.get("ares_output") and case.get("vita_pre_output"):
                set_simulation_status(case_id, "completed")
                return

            raise Exception("Simulazione incompleta: output pre-ospedalieri mancanti")

        if case.get("path_type") == "full":
            scenario_hospital_input = None

            for scenario in get_demo_scenarios():
                if scenario["demo_key"] == case.get("demo_key"):
                    scenario_hospital_input = scenario.get("hospital_input")
                    break

            if scenario_hospital_input:
                send_hospital_data(case_id, scenario_hospital_input)
                time.sleep(2)

                trigger_athena(case_id)
                time.sleep(2)

                trigger_vita_final(case_id)
                time.sleep(2)

                case = wait_for_case_fields(
                    case_id,
                    ["clara_output", "ares_output", "vita_pre_output", "hospital_input", "athena_output", "vita_final_output"]
                )
                if not case:
                    return

                if (
                    case.get("clara_output")
                    and case.get("ares_output")
                    and case.get("vita_pre_output")
                    and case.get("hospital_input")
                    and case.get("athena_output")
                    and case.get("vita_final_output")
                ):
                    set_simulation_status(case_id, "completed")
                else:
                    raise Exception("Simulazione incompleta: mancano uno o più output")

    except Exception as e:
        def updater(case):
            case["simulation_status"] = "error"
            case["simulation_error"] = str(e)
        update_case(case_id, updater)
        
def start_case_simulation_if_needed(case):
    case_id = case["case_id"]
    status = case.get("simulation_status", "idle")

    if status != "idle":
        return False

    def updater(c):
        c["phase"] = "pre_hospital"
        c["current_step"] = "clara_pending"
        c["simulation_status"] = "running"
        c["process_started"] = True
        c["manual_priority"] = None
        c.pop("simulation_error", None)

    update_case(case_id, updater)

    thread = threading.Thread(target=run_case_simulation, args=(case_id,), daemon=True)
    thread.start()
    return True


def ensure_and_start_demo_cases():
    demo_cases = ensure_demo_cases()

    for case in demo_cases:
        start_case_simulation_if_needed(case)

    refreshed_cases = []
    for case in ensure_demo_cases():
        refreshed_cases.append(case)

    return refreshed_cases

def get_demo_scenarios():
    return [
        {
            "demo_key": "demo_rosso",
            "path_type": "full",
            "payload": {
                "patient_identification": {
                    "name": "Giovanni Verdi",
                    "age": 72,
                    "sex": "M",
                    "patient_id": "AMB-2026-01001"
                },
                "operator_input": {
                    "main_symptom": "dolore toracico",
                    "symptom_onset": "2026-04-09T10:00:00",
                    "known_conditions": ["ipertensione"],
                    "allergies": ["nessuna nota"],
                    "chronic_medications": ["ramipril"]
                },
                "real_time_vitals": {
                    "blood_pressure_mmHg": "85/55",
                    "heart_rate_bpm": 128,
                    "spo2_percent": 87
                }
            },
            "hospital_input": {
                "advanced_ecg": "sopraslivellamento ST in derivazioni anteriori",
                "blood_gas": "ipossiemia lieve",
                "cbc": "nella norma",
                "reports": ["quadro compatibile con STEMI"],
                "troponin": "elevata"
            }
        },
        {
            "demo_key": "demo_giallo",
            "path_type": "full",
            "payload": {
                "patient_identification": {
                    "name": "Marco Bianchi",
                    "age": 55,
                    "sex": "M",
                    "patient_id": "AMB-2026-01003"
                },
                "operator_input": {
                    "main_symptom": "dispnea",
                    "symptom_onset": "2026-04-09T10:15:00",
                    "known_conditions": ["asma"],
                    "allergies": ["nessuna"],
                    "chronic_medications": ["salbutamolo"]
                },
                "real_time_vitals": {
                    "blood_pressure_mmHg": "110/70",
                    "heart_rate_bpm": 105,
                    "spo2_percent": 93
                }
            },
            "hospital_input": {
                "advanced_ecg": "senza segni di ischemia acuta",
                "blood_gas": "lieve ipossiemia",
                "cbc": "nella norma",
                "reports": ["quadro compatibile con riacutizzazione respiratoria"],
                "troponin": "negativa"
            }
        },
        {
            "demo_key": "demo_verde",
            "path_type": "full",
            "payload": {
                "patient_identification": {
                    "name": "Luca Rossi",
                    "age": 34,
                    "sex": "M",
                    "patient_id": "AMB-2026-01007"
                },
                "operator_input": {
                    "main_symptom": "frattura scomposta avambraccio destro",
                    "symptom_onset": "2026-04-09T09:30:00",
                    "known_conditions": [],
                    "allergies": ["nessuna"],
                    "chronic_medications": []
                },
                "real_time_vitals": {
                    "blood_pressure_mmHg": "125/80",
                    "heart_rate_bpm": 96,
                    "spo2_percent": 99
                }
            },
            "hospital_input": {
                "x_ray": "frattura scomposta diafisaria radio-ulna destra",
                "pain_assessment": "dolore severo locale senza deficit neurovascolare",
                "cbc": "nella norma",
                "reports": ["trauma ortopedico stabile, indicazione a riduzione e immobilizzazione"],
                "orthopedic_note": "arto perfuso, sensibilità conservata, necessario trattamento in ambiente ospedaliero"
            }
        },
        {
            "demo_key": "demo_bianco",
            "path_type": "pre_hospital",
            "payload": {
                "patient_identification": {
                    "name": "Andrea Neri",
                    "age": 24,
                    "sex": "M",
                    "patient_id": "AMB-2026-01011"
                },
                "operator_input": {
                    "main_symptom": "lieve contusione caviglia dopo distorsione",
                    "symptom_onset": "2026-04-09T11:10:00",
                    "known_conditions": [],
                    "allergies": ["nessuna"],
                    "chronic_medications": []
                },
                "real_time_vitals": {
                    "blood_pressure_mmHg": "122/78",
                    "heart_rate_bpm": 72,
                    "spo2_percent": 99
                }
            },
            "hospital_input": None
        }
    ]


def get_demo_scenario_by_key(demo_key):
    for scenario in get_demo_scenarios():
        if scenario["demo_key"] == demo_key:
            return scenario
    return None

def create_demo_case_from_scenario(scenario):
    case_id = f"CASE-{uuid.uuid4().hex[:8].upper()}"
    body = scenario["payload"]

    case_payload = {
        "case_id": case_id,
        "created_at": now_iso(),
        "updated_at": None,
        "phase": None,
        "current_step": None,
        "raw_input": body,
        "clara_output": None,
        "ares_output": None,
        "vita_pre_output": None,
        "hospital_input": None,
        "athena_output": None,
        "vita_final_output": None,
        "demo_key": scenario["demo_key"],
        "path_type": scenario["path_type"],
        "simulation_status": "idle",
        "process_started": False,
        "manual_priority": None,
        "schema_version": CASE_SCHEMA_VERSION,
        "timeline": {
            "created_at": now_iso(),
            "clara_at": None,
            "ares_at": None,
            "vita_pre_at": None,
            "hospital_at": None,
            "athena_at": None,
            "vita_final_at": None
        }
    }

    save_case(case_id, case_payload)
    return case_payload


def reset_demo_case_if_pristine(case, scenario):
    if case.get("schema_version") == CASE_SCHEMA_VERSION and case.get("process_started"):
        return case

    case["updated_at"] = None
    case["phase"] = None
    case["current_step"] = None
    case["raw_input"] = scenario["payload"]
    case["clara_output"] = None
    case["ares_output"] = None
    case["vita_pre_output"] = None
    case["hospital_input"] = None
    case["athena_output"] = None
    case["vita_final_output"] = None
    case["path_type"] = scenario["path_type"]
    case["simulation_status"] = "idle"
    case["process_started"] = False
    case["manual_priority"] = None
    case["schema_version"] = CASE_SCHEMA_VERSION
    case.pop("simulation_error", None)
    case["timeline"] = {
        "created_at": case.get("created_at") or now_iso(),
        "clara_at": None,
        "ares_at": None,
        "vita_pre_at": None,
        "hospital_at": None,
        "athena_at": None,
        "vita_final_at": None
    }
    save_case(case["case_id"], case)
    return case

def ensure_demo_cases():
    all_cases = get_all_cases()
    existing_by_demo_key = {}

    for case_id, case in all_cases.items():
        demo_key = case.get("demo_key")
        if demo_key:
            existing_by_demo_key[demo_key] = case

    demo_cases = []

    for scenario in get_demo_scenarios():
        demo_key = scenario["demo_key"]

        if demo_key in existing_by_demo_key:
            demo_cases.append(reset_demo_case_if_pristine(existing_by_demo_key[demo_key], scenario))
        else:
            new_case = create_demo_case_from_scenario(scenario)
            demo_cases.append(new_case)

    return demo_cases

def save_agent_output(case_id, field_name, payload, next_step=None, phase=None):
    case = get_case(case_id)
    if not case:
        return False

    case[field_name] = payload
    case["updated_at"] = now_iso()

    if next_step:
        case["current_step"] = next_step

    if phase:
        case["phase"] = phase

    case.setdefault("timeline", {})
    timeline_field_map = {
        "clara_output": "clara_at",
        "ares_output": "ares_at",
        "vita_pre_output": "vita_pre_at",
        "hospital_input": "hospital_at",
        "athena_output": "athena_at",
        "vita_final_output": "vita_final_at",
    }

    timeline_key = timeline_field_map.get(field_name)
    if timeline_key:
        case["timeline"][timeline_key] = get_simulated_timeline_timestamp(case, timeline_key)

    save_case(case_id, case)
    return True

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "emergency-ai-backend"
    }), 200


@app.route("/avatar-model")
def avatar_model():
        model_path = os.path.join(app.root_path, "static", "models", "model.glb")
        return send_file(model_path, mimetype="model/gltf-binary")


@app.route("/debug-avatar-path")
def debug_avatar_path():
    model_path = os.path.join(app.root_path, "static", "models", "model.glb")
    return {
        "root_path": app.root_path,
        "model_path": model_path,
        "exists": os.path.exists(model_path)
    }


@app.route("/api/test-storage", methods=["GET"])
def test_storage():
    return jsonify({
        "status": "ok",
        "message": "storage importato correttamente"
    }), 200


@app.route("/api/cases", methods=["POST"])
def create_case():
    body = request.get_json(silent=True) or {}

    case_id = f"CASE-{uuid.uuid4().hex[:8].upper()}"

    case_payload = {
        "case_id": case_id,
        "created_at": now_iso(),
        "updated_at": None,
        "timeline": {
            "created_at": now_iso(),
            "clara_at": None,
            "ares_at": None,
            "vita_pre_at": None,
            "hospital_at": None,
            "athena_at": None,
            "vita_final_at": None
        },
        "phase": None,
        "current_step": None,
        "raw_input": body,
        "clara_output": None,
        "ares_output": None,
        "vita_pre_output": None,
        "hospital_input": None,
        "athena_output": None,
        "vita_final_output": None,
        "simulation_status": "idle",
        "process_started": False,
        "manual_priority": None,
        "schema_version": CASE_SCHEMA_VERSION,
    }

    save_case(case_id, case_payload)

    return jsonify({
        "status": "success",
        "case_id": case_id,
        "current_step": "clara_pending",
        "message": "Caso creato correttamente"
    }), 201


@app.route("/api/cases", methods=["GET"])
def list_cases():
    all_cases = get_all_cases()
    return jsonify({
        "status": "success",
        "count": len(all_cases),
        "cases": all_cases
    }), 200


@app.route("/api/demo-cases", methods=["GET"])
def api_demo_cases():
    demo_cases = ensure_demo_cases()
    return jsonify({
        "status": "success",
        "count": len(demo_cases),
        "cases": demo_cases
    }), 200


@app.route("/api/cases/<case_id>", methods=["GET"])
def read_case(case_id):
    case_data = get_case(case_id)

    if not case_data:
        return jsonify({
            "status": "error",
            "message": "Caso non trovato"
        }), 404

    if case_data.get("demo_key"):
        scenario = get_demo_scenario_by_key(case_data.get("demo_key"))
        if scenario:
            case_data = reset_demo_case_if_pristine(case_data, scenario)

    return jsonify(case_data), 200


@app.route("/api/cases/<case_id>/start", methods=["POST"])
def start_case(case_id):
    case_data = get_case(case_id)

    if not case_data:
        return jsonify({
            "status": "error",
            "message": "Caso non trovato"
        }), 404

    force_restart = bool((request.get_json(silent=True) or {}).get("force_restart"))

    if force_restart:
        case_data = reset_case_for_restart(case_id)
        if not case_data:
            return jsonify({
                "status": "error",
                "message": "Impossibile resettare il caso"
            }), 500

    started = start_case_simulation_if_needed(case_data)
    refreshed_case = get_case(case_id)

    if not started:
        return jsonify({
            "status": "error",
            "message": "Simulazione non avviata: caso non in stato idle",
            "simulation_status": refreshed_case.get("simulation_status") if refreshed_case else None
        }), 409

    return jsonify({
        "status": "success",
        "message": "Simulazione avviata",
        "case_id": case_id,
        "simulation_status": refreshed_case.get("simulation_status") if refreshed_case else "running"
    }), 202


@app.route("/api/cases/<case_id>/priority", methods=["POST"])
def set_case_priority(case_id):
    case_data = get_case(case_id)

    if not case_data:
        return jsonify({
            "status": "error",
            "message": "Caso non trovato"
        }), 404

    body = request.get_json(silent=True) or {}
    selected_priority = str(body.get("priority", "")).strip().upper()

    if selected_priority and selected_priority not in MANUAL_PRIORITY_STYLES:
        return jsonify({
            "status": "error",
            "message": "Priorita non valida"
        }), 400

    def updater(case):
        case["manual_priority"] = selected_priority or None

    updated_case = update_case(case_id, updater)

    return jsonify({
        "status": "success",
        "message": "Priorita del caso aggiornata",
        "manual_priority": updated_case.get("manual_priority") if updated_case else (selected_priority or None)
    }), 200


@app.route("/api/cases/<case_id>/avatar-chat", methods=["POST"])
def avatar_chat(case_id):
    body = request.get_json(silent=True) or {}
    question = body.get("question", "").strip()

    if not question:
        return jsonify({"error": "Domanda mancante"}), 400

    case = get_case(case_id)
    if not case:
        return jsonify({"error": "Caso non trovato"}), 404

    clara = case.get("clara_output") or {}
    ares = case.get("ares_output") or {}
    vita_pre = case.get("vita_pre_output") or {}
    hospital_input = case.get("hospital_input") or {}
    athena = case.get("athena_output") or {}
    vita_final = case.get("vita_final_output") or {}

    case_context = {
        "case_id": case.get("case_id"),
        "phase": case.get("phase"),
        "current_step": case.get("current_step"),
        "created_at": case.get("created_at"),
        "updated_at": case.get("updated_at"),
        "clara_output": clara,
        "ares_output": ares,
        "vita_pre_output": vita_pre,
        "hospital_input": hospital_input,
        "athena_output": athena,
        "vita_final_output": vita_final
    }

    system_prompt = """
Sei EVAN, un avatar clinico avanzato integrato in una dashboard di emergenza.

Il tuo ruolo è integrare e interpretare l'intero caso come farebbe un medico esperto, usando:
- CLARA per dati clinici e parametri vitali
- ARES per pre-triage e stratificazione del rischio
- VITA per la sintesi clinica
- ATHENA per il ragionamento diagnostico

Obiettivo:
- rispondere in modo clinicamente rigoroso, naturale e aderente alla domanda reale dell'utente
- mantenere varietà di stile e formulazione tra una risposta e l'altra
- restare sempre dentro il perimetro clinico del caso attuale

Regole obbligatorie:
- usa esclusivamente i dati del caso forniti
- non inventare esami, diagnosi, tempi o informazioni mancanti
- se un dato non è disponibile, dichiaralo chiaramente
- rispondi sempre in italiano
- mantieni tono professionale, medico e chiaro
- non usare sempre la stessa apertura, la stessa chiusura o la stessa struttura
- adatta lunghezza, taglio e organizzazione della risposta alla domanda fatta
- se la domanda è diretta, rispondi in modo diretto
- se la domanda richiede ragionamento, spiega il ragionamento in modo scorrevole
- se la domanda richiede sintesi, non trasformarla in una risposta lunga
- se utile, puoi organizzare la risposta in paragrafi brevi o punti, ma solo quando serve davvero
- evita schemi rigidi ripetuti in ogni risposta

Priorità cliniche:
- distingui tra dati oggettivi, interpretazioni e ipotesi
- evidenzia instabilità, rischio, priorità e implicazioni operative quando rilevanti
- se l'utente chiede una spiegazione semplice, semplifica il linguaggio ma non il contenuto clinico

Non uscire mai dal contesto del caso clinico attuale.
"""

    user_prompt = f"""
Caso clinico attuale:

{json.dumps(case_context, ensure_ascii=False, indent=2)}

Domanda dell'utente:
{question}

Istruzioni di risposta:
- rispondi esattamente alla domanda, senza usare sempre lo stesso formato
- mantieni il focus sul caso attuale
- se la domanda è ampia, costruisci una risposta completa ma scorrevole
- se la domanda è specifica, resta focalizzato su quel punto
- evita formule ripetitive e risposte standardizzate
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7
        )

        answer = response.choices[0].message.content.strip()

        return jsonify({
            "case_id": case_id,
            "answer": answer
        }), 200

    except Exception as e:
        return jsonify({
            "error": f"Errore nella generazione della risposta avatar: {str(e)}"
        }), 500


@app.route("/api/cases/<case_id>/avatar-speech", methods=["POST"])
def avatar_speech(case_id):
    case = get_case(case_id)
    if not case:
        return jsonify({"error": "Caso non trovato"}), 404

    if not os.getenv("OPENAI_API_KEY"):
        return jsonify({"error": "OPENAI_API_KEY non configurata sul backend"}), 500

    body = request.get_json(silent=True) or {}
    text = str(body.get("text", "")).strip()

    if not text:
        return jsonify({"error": "Testo mancante"}), 400

    normalized_text = normalize_tts_text(text)

    if not normalized_text:
        return jsonify({"error": "Testo non valido per la sintesi"}), 400

    normalized_text = normalized_text[:4000]
    ensure_avatar_audio_dir()

    cache_key = hashlib.sha256(
        f"{case_id}|{TTS_MODEL}|{TTS_VOICE}|{normalized_text}".encode("utf-8")
    ).hexdigest()[:20]
    audio_filename = f"{case_id}-{cache_key}.wav"
    audio_path = os.path.join(AVATAR_AUDIO_DIR, audio_filename)
    audio_url = f"/static/generated/avatar_audio/{audio_filename}"
    json_filename = f"{case_id}-{cache_key}.json"
    json_path = os.path.join(AVATAR_AUDIO_DIR, json_filename)

    try:
        if not os.path.exists(audio_path):
            audio_response = client.audio.speech.create(
                model=TTS_MODEL,
                voice=TTS_VOICE,
                input=normalized_text,
                instructions=(
                    "Leggi in italiano in modo clinico, fluido e naturale. "
                    "Pronuncia con chiarezza numeri, unita di misura, sigle mediche e parametri vitali."
                ),
                response_format="wav",
                timeout=TTS_TIMEOUT_SECONDS,
            )
            audio_bytes = audio_response.read()
            with open(audio_path, "wb") as file:
                file.write(audio_bytes)
        else:
            with open(audio_path, "rb") as file:
                audio_bytes = file.read()

        duration_seconds = estimate_audio_duration_seconds(audio_bytes)

        cue_payload = generate_rhubarb_cues(
            audio_path=audio_path,
            text=normalized_text,
            duration_seconds=duration_seconds,
        )

        with open(json_path, "w", encoding="utf-8") as file:
            json.dump(cue_payload, file, ensure_ascii=False, indent=2)

        return jsonify({
            "audio_url": audio_url,
            "duration_seconds": duration_seconds,
            "cue_provider": cue_payload["provider"],
            "mouth_cues": cue_payload["cues"],
        }), 200
    except Exception as e:
        return jsonify({
            "error": f"Errore nella generazione audio avatar: {str(e)}"
        }), 500


@app.route("/api/cases/<case_id>/agent-input/<agent_name>", methods=["GET"])
def get_agent_input(case_id, agent_name):
    case_data = get_case(case_id)

    if not case_data:
        return jsonify({
            "status": "error",
            "message": "Caso non trovato"
        }), 404

    payload = build_agent_input(case_data, agent_name)

    if payload is None:
        return jsonify({
            "status": "error",
            "message": "Agente non valido"
        }), 400

    return jsonify({
        "status": "success",
        "agent": agent_name,
        "input": payload
    }), 200


@app.route("/api/cases/<case_id>/clara", methods=["POST"])
def save_clara_output(case_id):
    body = request.get_json(silent=True) or {}

    updated = save_agent_output(
        case_id=case_id,
        field_name="clara_output",
        payload=body,
        next_step="ares_pending"
    )

    if not updated:
        return jsonify({
            "status": "error",
            "message": "Caso non trovato"
        }), 404

    return jsonify({
        "status": "success",
        "message": "Output CLARA salvato",
        "next_step": "ares_pending"
    }), 200


@app.route("/api/cases/<case_id>/ares", methods=["POST"])
def save_ares_output(case_id):
    body = request.get_json(silent=True) or {}
    case = get_case(case_id)

    if not case:
        return jsonify({
            "status": "error",
            "message": "Caso non trovato"
        }), 404

    body = normalize_demo_agent_payload(case, "ares_output", body)

    updated = save_agent_output(
        case_id=case_id,
        field_name="ares_output",
        payload=body,
        next_step="vita_pre_pending"
    )

    if not updated:
        return jsonify({
            "status": "error",
            "message": "Caso non trovato"
        }), 404

    return jsonify({
        "status": "success",
        "message": "Output ARES salvato",
        "next_step": "vita_pre_pending"
    }), 200

@app.route("/api/cases/<case_id>/vita/pre", methods=["POST"])
def save_vita_pre_output(case_id):
    body = request.get_json(silent=True) or {}
    case = get_case(case_id)

    if not case:
        return jsonify({
            "status": "error",
            "message": "Caso non trovato"
        }), 404

    body = normalize_demo_agent_payload(case, "vita_pre_output", body)
    next_step = "completed" if case.get("path_type") == "pre_hospital" else "waiting_hospital_data"

    updated = save_agent_output(
        case_id=case_id,
        field_name="vita_pre_output",
        payload=body,
        next_step=next_step
    )

    if not updated:
        return jsonify({
            "status": "error",
            "message": "Caso non trovato"
        }), 404

    return jsonify({
        "status": "success",
        "message": "Output VITA fase 1 salvato",
        "next_step": next_step
    }), 200

@app.route("/api/cases/<case_id>/hospital-data", methods=["POST"])
def add_hospital_data(case_id):
    body = request.get_json(silent=True) or {}

    updated = save_agent_output(
        case_id=case_id,
        field_name="hospital_input",
        payload=body,
        next_step="athena_pending",
        phase="hospital"
    )

    if not updated:
        return jsonify({
            "status": "error",
            "message": "Caso non trovato"
        }), 404

    return jsonify({
        "status": "success",
        "message": "Dati ospedalieri acquisiti",
        "next_step": "athena_pending"
    }), 200

@app.route("/api/cases/<case_id>/athena", methods=["POST"])
def save_athena_output(case_id):
    body = request.get_json(silent=True) or {}

    updated = save_agent_output(
        case_id=case_id,
        field_name="athena_output",
        payload=body,
        next_step="vita_final_pending"
    )

    if not updated:
        return jsonify({
            "status": "error",
            "message": "Caso non trovato"
        }), 404

    return jsonify({
        "status": "success",
        "message": "Output ATHENA salvato",
        "next_step": "vita_final_pending"
    }), 200

@app.route("/api/cases/<case_id>/vita/final", methods=["POST"])
def save_vita_final_output(case_id):
    body = request.get_json(silent=True) or {}

    updated = save_agent_output(
        case_id=case_id,
        field_name="vita_final_output",
        payload=body,
        next_step="completed"
    )

    if not updated:
        return jsonify({
            "status": "error",
            "message": "Caso non trovato"
        }), 404

    return jsonify({
        "status": "success",
        "message": "Output VITA fase 2 salvato",
        "next_step": "completed"
    }), 200

@app.route("/dashboard/<case_id>")
def dashboard(case_id):
    case = get_case(case_id)

    if not case:
        return {"error": "Case not found"}, 404

    if case.get("demo_key"):
        scenario = get_demo_scenario_by_key(case.get("demo_key"))
        if scenario:
            case = reset_demo_case_if_pristine(case, scenario)

    return render_template(
        "dashboard.html",
        case=case,
        auto_start_enabled=False
    )

@app.route("/")
def index():
    demo_cases = ensure_demo_cases()
    all_cases = {case["case_id"]: case for case in demo_cases}
    case_cards = []

    for case_id, case in all_cases.items():
        clara_output = case.get("clara_output") or {}
        ares_output = case.get("ares_output") or {}
        vita_pre_output = case.get("vita_pre_output") or {}
        vita_final_output = case.get("vita_final_output") or {}

        patient_record = clara_output.get("patient_record") or {}
        patient_identification = patient_record.get("patient_identification") or {}

        raw_patient = case.get("raw_input", {}).get("patient_identification", {})
        operator_input = case.get("raw_input", {}).get("operator_input", {})
        main_complaint = patient_record.get("main_complaint") or {}

        patient_name = (
            patient_identification.get("name")
            or raw_patient.get("name")
            or "Paziente non disponibile"
        )

        age = (
            patient_identification.get("age")
            or raw_patient.get("age")
            or "N/D"
        )

        sex = (
            patient_identification.get("sex")
            or raw_patient.get("sex")
            or "N/D"
        )

        main_symptom = (
            main_complaint.get("main_symptom")
            or operator_input.get("main_symptom")
            or "Non disponibile"
        )

        priority = case.get("manual_priority")
        suggested_priority = normalize_priority_code(ares_output.get("priority_code"))
        if case.get("demo_key") == "demo_bianco" and suggested_priority == "VERDE":
            suggested_priority = "BIANCO"
        risk_level = ares_output.get("immediate_risk_level")

        summary = (
            vita_final_output.get("dashboard_summary")
            or vita_pre_output.get("dashboard_summary")
            or clara_output.get("dashboard_payload", {}).get("alert")
            or f"Caso aperto per {main_symptom.lower()}."
        )

        process_started = case.get("process_started", False)
        severity_class = "neutral"

        if priority == "ROSSO":
            severity_class = "red"
        elif priority == "GIALLO":
            severity_class = "yellow"
        elif priority == "VERDE":
            severity_class = "green"
        elif priority == "BIANCO":
            severity_class = "white"

        display_priority = "-"

        if severity_class == "red":
            display_priority = "critico"
        elif severity_class == "yellow":
            display_priority = "medio"
        elif severity_class == "green":
            display_priority = "stabile"
        elif severity_class == "white":
            display_priority = "minore"

        current_step = case.get("current_step") or "-"
        if current_step == "completed":
            current_step = "completato"

        phase = case.get("phase") or "-"
        updated_at = case.get("updated_at") or "-"

        case_cards.append({
            "case_id": case_id,
            "patient_name": patient_name,
            "age": age,
            "sex": sex,
            "main_symptom": main_symptom,
            "phase": phase,
            "current_step": current_step,
            "priority": priority,
            "suggested_priority": suggested_priority,
            "risk_level": risk_level,
            "summary": summary,
            "severity_class": severity_class,
            "updated_at": updated_at,
            "display_priority": display_priority,
            "simulation_status": get_status_label(case.get("simulation_status", "idle")),
            "raw_simulation_status": case.get("simulation_status", "idle"),
        })

    return render_template("index.html", cases=case_cards)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False, use_reloader=False)
