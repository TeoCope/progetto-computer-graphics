# Questo dizionario mappa il nome "leggibile" di un landmark anatomico
# (es. la punta della testa, un capezzolo, un polso...) all'indice del
# vertice corrispondente nella mesh del corpo SMPL, che è composta da
# 6890 vertici in totale. Questi indici sono stati individuati a mano
# (o presi da dataset come CAESAR) una volta per tutte sul modello SMPL
# "a riposo" e vengono poi riusati per calcolare le misure antropometriche
# (lunghezze e circonferenze) su qualunque corpo generato da SMPL, perché
# la mesh ha sempre la stessa topologia (stessi indici = stesse zone del corpo).
SMPL_LANDMARK_INDICES = {"HEAD_TOP": 412,           # vertice sulla sommità della testa
                    "HEAD_LEFT_TEMPLE": 166,         # vertice sulla tempia sinistra
                    "NECK_ADAM_APPLE": 3050,         # vertice sul "pomo d'Adamo" (gola)
                    "LEFT_HEEL": 3458,               # vertice sul tallone sinistro
                    "RIGHT_HEEL": 6858,              # vertice sul tallone destro
                    "LEFT_NIPPLE": 3042,
                    "RIGHT_NIPPLE": 6489,

                    "SHOULDER_TOP": 3068,
                    "INSEAM_POINT": 3149,
                    "BELLY_BUTTON": 3501,
                    "BACK_BELLY_BUTTON": 3022,
                    "CROTCH": 1210,
                    "PUBIC_BONE": 3145,
                    "RIGHT_WRIST": 5559,
                    "LEFT_WRIST": 2241,
                    "RIGHT_BICEP": 4855,
                    "RIGHT_FOREARM": 5197,
                    "LEFT_SHOULDER": 3011,
                    "RIGHT_SHOULDER": 6470,
                    "LOW_LEFT_HIP": 3134,
                    "LEFT_THIGH": 947,
                    "LEFT_CALF": 1103,
                    "LEFT_ANKLE": 3325,
                    "LEFT_ELBOW": 1643,

                    "BUTTHOLE": 3119,

                    # Questi 4 landmark seguono la nomenclatura del dataset CAESAR
                    # (non il naming "generico" usato sopra) e sono stati aggiunti
                    # perché servono a spezzare la misura del braccio in più tratti
                    # (spalla-gomito, gomito-polso) invece di misurarla tutta d'un
                    # pezzo: Cervicale = vertebra alla base del collo, Acromion =
                    # punta della spalla, Humeral Lateral Epicondyle = gomito,
                    # Ulnar Styloid = polso.
                    "Cervicale": 829,
                    'Rt. Acromion': 5342,
                    'Rt. Humeral Lateral Epicn': 5090,
                    'Rt. Ulnar Styloid': 5520,
                    }

# Voce "derivata": non è un singolo indice di vertice ma una tupla con i due
# indici già definiti sopra (tallone sinistro, tallone destro). Viene creata
# così, dopo la definizione del dizionario, perché serve un punto "medio/doppio"
# usato ad esempio per misurare l'altezza totale del corpo (dalla testa ai
# talloni) o la circonferenza dell'anca alla massima altezza.
SMPL_LANDMARK_INDICES["HEELS"] = (SMPL_LANDMARK_INDICES["LEFT_HEEL"],
                                  SMPL_LANDMARK_INDICES["RIGHT_HEEL"])


# Stesso concetto del dizionario precedente, ma per la mesh SMPLX, che ha una
# topologia diversa (10475 vertici invece di 6890): per questo gli indici sono
# diversi anche per landmark con lo stesso nome. SMPLX qui definisce meno
# landmark rispetto a SMPL (mancano ad es. "BUTTHOLE" e i landmark in stile
# CAESAR per il braccio), perché nel progetto le misure per SMPLX sono un
# sottoinsieme di quelle disponibili per SMPL (vedi measurement_definitions.py).
SMPLX_LANDMARK_INDICES = {"HEAD_TOP": 8976,          # vertice sulla sommità della testa
                    "HEAD_LEFT_TEMPLE": 1980,         # vertice sulla tempia sinistra
                    "NECK_ADAM_APPLE": 8940,          # vertice sul "pomo d'Adamo" (gola)
                    "LEFT_HEEL": 8847,                # vertice sul tallone sinistro
                    "RIGHT_HEEL": 8635,               # vertice sul tallone destro
                    "LEFT_NIPPLE": 3572,
                    "RIGHT_NIPPLE": 8340,

                    "SHOULDER_TOP": 5616,
                    "INSEAM_POINT": 5601,
                    "BELLY_BUTTON": 5939,
                    "BACK_BELLY_BUTTON": 5941,
                    "CROTCH": 3797,
                    "PUBIC_BONE": 5949,
                    "RIGHT_WRIST": 7449,
                    "LEFT_WRIST": 4823,
                    "RIGHT_BICEP": 6788, 
                    "RIGHT_FOREARM": 7266,
                    "LEFT_SHOULDER": 4442,
                    "RIGHT_SHOULDER": 7218, 
                    "LOW_LEFT_HIP": 4112, 
                    "LEFT_THIGH": 3577,
                    "LEFT_CALF": 3732,
                    "LEFT_ANKLE": 5880
                    }

# Come per SMPL: voce "derivata" aggiunta dopo il dizionario, tupla con gli
# indici dei due talloni (sinistro, destro), usata per le misure di altezza.
SMPLX_LANDMARK_INDICES["HEELS"] = (SMPLX_LANDMARK_INDICES["LEFT_HEEL"],
                                  SMPLX_LANDMARK_INDICES["RIGHT_HEEL"])