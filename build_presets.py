# ============================================================================
# SCRIPT build_presets.py
# Script "offline" (da lanciare a mano, non fa parte della GUI) che calcola
# le betas per ciascun preset di avatar definito in presets.json.
#
# Per ogni preset: carica il modello SMPL del genere giusto, esegue il
# fitting L-BFGS-B sulle misure antropometriche target (vedi fitting.py),
# poi salva nel JSON le betas trovate e le misure effettivamente ottenute,
# stampando l'errore per ogni misura. La GUI (main.py) leggerà direttamente
# le betas precalcolate, senza dover rifare l'ottimizzazione.
#
# Uso:  python build_presets.py [--force]
#   --force  ricalcola anche i preset che hanno già le betas nel JSON
# ============================================================================
import os
import sys
import argparse

import numpy as np

# Il modulo "measure" vive nella sottocartella del sottoprogetto SMPL-Anthropometry
sys.path.append(os.path.join(os.path.dirname(__file__), "SMPL-Anthropometry-master"))
from measure import MeasureBody

from fitting import (
    fit_betas_to_measurements,
    compute_measurements,
    load_presets,
    save_presets,
    DEFAULT_PRESETS_PATH,
)


def fit_for_model_type(model_type, gender, targets, models_root='data/body_models'):
    # Esegue il fitting delle misure target per un dato tipo di modello
    # (smpl o smplx) e genere. Ritorna (betas, misure_ottenute, risultato_scipy).
    from smplx import SMPL, SMPLX

    model_class = {'smpl': SMPL, 'smplx': SMPLX}[model_type]
    model = model_class(os.path.join(models_root, model_type), gender=gender)

    measurer = MeasureBody(model_type)
    measurer.gender = gender.upper()

    betas, res = fit_betas_to_measurements(model, measurer, targets)
    achieved = compute_measurements(model, measurer, betas, targets.keys())
    return betas, achieved, res


def print_fit_report(targets, betas, achieved, res):
    print(f"Betas:  " + ", ".join(f"{b:+.2f}" for b in betas))
    print(f"Loss finale: {res.fun:.4f}")
    for k, target_val in targets.items():
        err = achieved.get(k, float('nan')) - target_val
        print(f"  {k:28s} target {target_val:6.1f} cm -> ottenuto {achieved.get(k, float('nan')):6.1f} cm (errore {err:+.2f} cm)")


def build_preset(preset, models_root='data/body_models', force=False):
    # Esegue il fitting per un singolo preset e riempie i campi "betas" e
    # "achieved_measurements" del dizionario. Oltre al modello principale del
    # preset (di norma smpl, usato dalla GUI Python), calcola anche le betas
    # per SMPL-X ("betas_smplx"): il pacchetto ufficiale "SMPL-X for Unity"
    # usa quel modello, e gli spazi di forma di SMPL e SMPL-X NON sono
    # compatibili tra loro, quindi servono due fit separati sulle stesse misure.
    # I fit già presenti nel JSON vengono saltati, a meno di --force.
    model_type = preset.get('model_type', 'smpl').lower()
    gender = preset.get('gender', 'neutral').lower()
    targets = preset['target_measurements']

    print(f"\n=== Preset '{preset['label']}' ({model_type}, {gender}) ===")
    print(f"Target: " + ", ".join(f"{k}={v:.1f}" for k, v in targets.items()))

    if preset.get('betas') is None or force:
        betas, achieved, res = fit_for_model_type(model_type, gender, targets, models_root)
        print_fit_report(targets, betas, achieved, res)
        preset['betas'] = [round(float(b), 4) for b in betas]
        preset['achieved_measurements'] = {k: round(float(v), 2) for k, v in achieved.items()}
    else:
        print("Betas già presenti, salto (usa --force per ricalcolare)")

    if model_type != 'smplx' and (preset.get('betas_smplx') is None or force):
        print(f"--- Variante SMPL-X (per Unity) ---")
        betas_x, achieved_x, res_x = fit_for_model_type('smplx', gender, targets, models_root)
        print_fit_report(targets, betas_x, achieved_x, res_x)
        preset['betas_smplx'] = [round(float(b), 4) for b in betas_x]
        preset['achieved_measurements_smplx'] = {k: round(float(v), 2) for k, v in achieved_x.items()}

    return preset


def main():
    parser = argparse.ArgumentParser(description="Calcola le betas dei preset di avatar")
    parser.add_argument('--force', action='store_true',
                        help='Ricalcola anche i preset con betas già presenti nel JSON')
    parser.add_argument('--presets', default=DEFAULT_PRESETS_PATH,
                        help='Percorso del file presets.json')
    args = parser.parse_args()

    presets = load_presets(args.presets)
    if not presets:
        print(f"Nessun preset trovato in {args.presets}")
        sys.exit(1)

    for preset in presets:
        build_preset(preset, force=args.force)

    save_presets(presets, args.presets)
    print(f"\nPreset salvati in {args.presets}")


if __name__ == '__main__':
    main()
