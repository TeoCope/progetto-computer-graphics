# ============================================================================
# SCRIPT avatar_server.py
# Piccolo server HTTP locale che espone all'applicazione Unity la STESSA
# logica di fitting antropometrico della GUI Python: misuratore geometrico
# MeasureBody (piani di taglio + convex hull) e ottimizzatore L-BFGS-B
# (funzioni condivise in fitting.py). In questo modo Unity può far definire
# l'avatar all'utente tramite le misure in cm, esattamente come nel main,
# senza duplicare o approssimare la logica in C#.
#
# Solo libreria standard (http.server): nessuna dipendenza aggiuntiva.
#
# Uso:  python avatar_server.py [--port 8077]
#
# Endpoints (request e response in JSON):
#   GET  /health   -> {"status": "ok", "models": [...]}
#   GET  /presets  -> contenuto di presets.json
#   POST /fit      -> {"model_type", "gender", "measurements": {chiave: cm},
#                      "initial_betas"?: [10]}
#                     risponde {"betas", "achieved_measurements",
#                               "max_error_cm", "loss"}
#   POST /measure  -> {"model_type", "gender", "betas": [10]}
#                     risponde {"measurements": {chiave: cm}}
# ============================================================================
import os
import sys
import json
import argparse
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "SMPL-Anthropometry-master"))
from measure import MeasureBody

from fitting import (
    fit_betas_to_measurements,
    compute_measurements,
    N_BETAS,
    DEFAULT_PRESETS_PATH,
)

# Le stesse misure disponibili come target nella GUI Python (main.py)
MEASUREMENT_KEYS = [
    "height", "chest circumference", "waist circumference", "hip circumference",
    "inside leg height", "arm right length",
    "neck circumference", "head circumference", "shoulder breadth",
]

MODEL_TYPES = ('smpl', 'smplx')
GENDERS = ('neutral', 'male', 'female')


class AvatarService:
    # Tiene in memoria i modelli smplx precaricati e i misuratori, e offre
    # le due operazioni usate dagli endpoint. Un lock serializza le richieste:
    # i MeasureBody sono oggetti con stato (verts/joints/gender) condivisi
    # tra i thread del server, quindi non sono thread-safe.
    def __init__(self, models_root='data/body_models'):
        from smplx import SMPL, SMPLX

        self.models = {}
        for model_type, model_class in (('smpl', SMPL), ('smplx', SMPLX)):
            for gender in GENDERS:
                print(f"Carico {model_type}-{gender}...")
                self.models[f'{model_type}-{gender}'] = model_class(
                    os.path.join(models_root, model_type), gender=gender)

        self.measurers = {mt: MeasureBody(mt) for mt in MODEL_TYPES}
        self.lock = threading.Lock()

    def _get(self, model_type, gender):
        model_type = (model_type or 'smplx').lower()
        gender = (gender or 'neutral').lower()
        key = f'{model_type}-{gender}'
        if key not in self.models:
            raise ValueError(f"Modello non disponibile: {key}")
        measurer = self.measurers[model_type]
        measurer.gender = gender.upper()
        return self.models[key], measurer

    def fit(self, model_type, gender, measurements, initial_betas=None):
        # Filtriamo come fa la GUI: contano solo le misure con valore > 0
        targets = {k: float(v) for k, v in (measurements or {}).items()
                   if k in MEASUREMENT_KEYS and float(v) > 0.0}
        if not targets:
            raise ValueError("Nessuna misura valida (> 0 cm); chiavi supportate: "
                             + ", ".join(MEASUREMENT_KEYS))
        if initial_betas is not None and len(initial_betas) != N_BETAS:
            raise ValueError(f"initial_betas deve avere {N_BETAS} valori")

        with self.lock:
            model, measurer = self._get(model_type, gender)
            betas, res = fit_betas_to_measurements(
                model, measurer, targets,
                initial_betas=np.array(initial_betas) if initial_betas else None)
            achieved = compute_measurements(model, measurer, betas, targets.keys())

        errors = [abs(achieved[k] - v) for k, v in targets.items() if k in achieved]
        return {
            'betas': [round(float(b), 4) for b in betas],
            'achieved_measurements': {k: round(float(v), 2) for k, v in achieved.items()},
            'max_error_cm': round(float(max(errors)), 2) if errors else 0.0,
            'loss': round(float(res.fun), 4),
        }

    def measure(self, model_type, gender, betas):
        if betas is None or len(betas) != N_BETAS:
            raise ValueError(f"betas deve avere {N_BETAS} valori")
        with self.lock:
            model, measurer = self._get(model_type, gender)
            m = compute_measurements(model, measurer, np.array(betas, dtype=float),
                                     MEASUREMENT_KEYS)
        return {'measurements': {k: round(float(v), 2) for k, v in m.items()}}


class AvatarRequestHandler(BaseHTTPRequestHandler):
    service = None  # Impostato in main() prima di avviare il server

    def _send_json(self, code, payload):
        body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == '/health':
            self._send_json(200, {'status': 'ok',
                                  'models': sorted(self.service.models.keys()),
                                  'measurement_keys': MEASUREMENT_KEYS})
        elif self.path == '/presets':
            if os.path.exists(DEFAULT_PRESETS_PATH):
                with open(DEFAULT_PRESETS_PATH, 'r', encoding='utf-8') as f:
                    self._send_json(200, json.load(f))
            else:
                self._send_json(404, {'error': 'presets.json non trovato'})
        else:
            self._send_json(404, {'error': f'Endpoint sconosciuto: {self.path}'})

    def do_POST(self):
        try:
            length = int(self.headers.get('Content-Length', 0))
            data = json.loads(self.rfile.read(length).decode('utf-8')) if length else {}
        except (ValueError, json.JSONDecodeError) as e:
            self._send_json(400, {'error': f'JSON non valido: {e}'})
            return

        try:
            if self.path == '/fit':
                result = self.service.fit(
                    data.get('model_type'), data.get('gender'),
                    data.get('measurements'), data.get('initial_betas'))
                self._send_json(200, result)
            elif self.path == '/measure':
                result = self.service.measure(
                    data.get('model_type'), data.get('gender'), data.get('betas'))
                self._send_json(200, result)
            else:
                self._send_json(404, {'error': f'Endpoint sconosciuto: {self.path}'})
        except ValueError as e:
            # Errore di input dell'utente: 400 con messaggio leggibile
            self._send_json(400, {'error': str(e)})
        except Exception as e:
            # Errore interno inatteso: 500, ma il server resta vivo
            self._send_json(500, {'error': f'{type(e).__name__}: {e}'})

    def log_message(self, format, *args):
        # Log compatto su stdout (il default stampa anche data/ora ridondanti)
        print(f"[{self.address_string()}] {format % args}")


def main():
    parser = argparse.ArgumentParser(
        description="Server locale di fitting antropometrico per l'app Unity")
    parser.add_argument('--port', type=int, default=8077)
    args = parser.parse_args()

    AvatarRequestHandler.service = AvatarService()

    server = ThreadingHTTPServer(('127.0.0.1', args.port), AvatarRequestHandler)
    print(f"\nServer di fitting avviato su http://localhost:{args.port}")
    print("Endpoints: GET /health, GET /presets, POST /fit, POST /measure")
    print("Ctrl+C per fermarlo.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer fermato.")


if __name__ == '__main__':
    main()
