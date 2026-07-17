# ============================================================================
# MODULO fitting.py
# Logica di "fitting" antropometrico condivisa tra la GUI (main.py) e lo
# script offline build_presets.py: date delle misure corporee target (in cm),
# trova i 10 parametri di forma (betas) del modello SMPL/SMPLX che le
# riproducono il più fedelmente possibile.
# ============================================================================
import copy
import json
import os

import numpy as np
import torch
from scipy.optimize import minimize

# Ogni beta può variare tra -5 e 5 (come gli slider della GUI), per evitare
# che l'ottimizzatore produca forme corporee irrealistiche
BETA_BOUNDS = (-5.0, 5.0)
N_BETAS = 10

# Percorso di default del file con i preset di avatar (relativo alla root del progetto)
DEFAULT_PRESETS_PATH = os.path.join(os.path.dirname(__file__), "presets.json")


def compute_measurements(model, measurer, betas, measurement_names,
                         expression=None, pose_params=None):
    # Esegue il forward pass del modello con i betas dati e calcola le misure
    # richieste sulla mesh risultante (con i piedi appoggiati a y=0, come fa
    # load_body_model in main.py, altrimenti le misure risulterebbero scorrette).
    # Ritorna un dizionario {nome_misura: valore_in_cm}.
    if expression is None:
        expression = torch.zeros(1, 10)
    if pose_params is None:
        pose_params = {}
    input_params = {k: v.reshape(1, -1) for k, v in copy.deepcopy(pose_params).items()}

    betas_t = torch.as_tensor(np.asarray(betas), dtype=torch.float32).reshape(1, N_BETAS)
    model_output = model(betas=betas_t, expression=expression, **input_params)
    verts = model_output.vertices[0].detach().numpy()
    joints = model_output.joints[0].detach().numpy()

    min_y = -verts[:, 1].min()
    measurer.verts = verts + np.array([0, min_y, 0])
    measurer.joints = joints + np.array([0, min_y, 0])
    measurer.measurements = {}
    measurer.measure(list(measurement_names))
    return dict(measurer.measurements)


def fit_betas_to_measurements(model, measurer, targets, initial_betas=None,
                              expression=None, pose_params=None, maxiter=30):
    # Cuore dell'algoritmo di fitting: cerca i betas che minimizzano la somma
    # degli scarti al quadrato tra le misure del corpo generato e i target
    # (errore quadratico, come in una regressione ai minimi quadrati).
    # Usa scipy L-BFGS-B con derivate stimate numericamente (eps), perché le
    # misure geometriche (piani di taglio + convex hull) non sono differenziabili.
    #
    # Parametri:
    #   model       -> istanza smplx (SMPL/SMPLX) del genere desiderato
    #   measurer    -> istanza MeasureBody corrispondente (con .gender già impostato)
    #   targets     -> dizionario {nome_misura: valore_target_in_cm}
    #   initial_betas -> punto di partenza (default: corpo medio, tutti zero)
    # Ritorna (betas_ottimizzati, risultato_scipy).
    if initial_betas is None:
        initial_betas = np.zeros(N_BETAS)

    def fit_loss(betas_np):
        measurements = compute_measurements(
            model, measurer, betas_np, targets.keys(),
            expression=expression, pose_params=pose_params,
        )
        loss = 0.0
        for k, target_val in targets.items():
            if k in measurements:
                loss += (measurements[k] - target_val) ** 2
        return loss

    bounds = [BETA_BOUNDS] * N_BETAS
    res = minimize(
        fit_loss,
        np.asarray(initial_betas, dtype=float),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': maxiter, 'ftol': 1e-4, 'eps': 0.1},
    )
    return res.x, res


def load_presets(path=DEFAULT_PRESETS_PATH):
    # Carica i preset di avatar dal file JSON. Ritorna la lista dei preset
    # (vuota se il file non esiste, così la GUI funziona anche senza preset).
    if not os.path.exists(path):
        return []
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('presets', [])


def save_presets(presets, path=DEFAULT_PRESETS_PATH):
    # Riscrive il file JSON dei preset (usato da build_presets.py per
    # salvare le betas calcolate accanto alle misure target).
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'presets': presets}, f, indent=2, ensure_ascii=False)
        f.write('\n')
