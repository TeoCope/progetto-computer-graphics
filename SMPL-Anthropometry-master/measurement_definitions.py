
# Importa i dizionari di landmark (nome landmark -> indice vertice mesh) e di
# giunti (indice/nome giunto) definiti negli altri due file "dati". Questo
# file li combina per definire COSA misurare e COME (lunghezza o
# circonferenza), mentre il calcolo vero e proprio della misura è fatto
# altrove dal Measurer.
from landmark_definitions import *
from joint_definitions import *

# Dizionario che associa una lettera "standard" (usata ad es. in tabelle di
# taglie/sartoria) al nome descrittivo della misura corrispondente usato nel
# resto del codice. Serve solo per avere etichette leggibili quando si
# presentano i risultati.
STANDARD_LABELS = {
        'A': 'head circumference',    # A -> circonferenza della testa
        'B': 'neck circumference',    # B -> circonferenza del collo
        'C': 'shoulder to crotch height',  # C -> altezza spalla-inguine
        'D': 'chest circumference',   # D -> circonferenza del torace
        'E': 'waist circumference',   # E -> circonferenza della vita
        'F': 'hip circumference',
        'G': 'wrist right circumference',
        'H': 'bicep right circumference',
        'I': 'forearm right circumference',
        'J': 'arm right length',
        'K': 'inside leg height',
        'L': 'thigh left circumference',
        'M': 'calf left circumference',
        'N': 'ankle left circumference',
        'O': 'shoulder breadth',
        'P': 'height'
    }


# Classe usata come semplice "enum": non viene istanziata, serve solo a dare
# un nome simbolico alle due stringhe possibili per il tipo di misura, invece
# di scrivere "circumference"/"length" a mano (con rischio di refusi) in giro
# per il codice.
class MeasurementType():
    CIRCUMFERENCE = "circumference"
    LENGTH = "length"


# Dizionario che associa ad ogni nome di misura il suo TIPO (lunghezza o
# circonferenza). Viene usato da Measurer.measure() per decidere se calcolare
# la misura come distanza tra due punti (measure_length) oppure come
# perimetro di una sezione del corpo (measure_circumference).
MEASUREMENT_TYPES = {
        "height": MeasurementType.LENGTH,                       # è una lunghezza
        "head circumference": MeasurementType.CIRCUMFERENCE,   # è una circonferenza
        "neck circumference": MeasurementType.CIRCUMFERENCE,
        "shoulder to crotch height": MeasurementType.LENGTH,
        "chest circumference": MeasurementType.CIRCUMFERENCE,
        "waist circumference": MeasurementType.CIRCUMFERENCE,
        "hip circumference": MeasurementType.CIRCUMFERENCE,

        "wrist right circumference": MeasurementType.CIRCUMFERENCE,
        "bicep right circumference": MeasurementType.CIRCUMFERENCE,
        "forearm right circumference": MeasurementType.CIRCUMFERENCE,
        "arm right length": MeasurementType.LENGTH,
        "arm left length":  MeasurementType.LENGTH,
        "inside leg height": MeasurementType.LENGTH,
        "thigh left circumference": MeasurementType.CIRCUMFERENCE,
        "calf left circumference": MeasurementType.CIRCUMFERENCE,
        "ankle left circumference": MeasurementType.CIRCUMFERENCE,
        "shoulder breadth": MeasurementType.LENGTH,

        "arm length (shoulder to elbow)": MeasurementType.LENGTH,
        "arm length (spine to wrist)": MeasurementType.LENGTH,
        "crotch height": MeasurementType.LENGTH,
        "Hip circumference max height": MeasurementType.LENGTH
    }

class SMPLMeasurementDefinitions():
    '''
    Definition of SMPL measurements.

    To add a new measurement:
    1. add it to the measurement_types dict and set the type:
       LENGTH or CIRCUMFERENCE
    2. depending on the type, define the measurement in LENGTHS or 
       CIRCUMFERENCES dict
       - LENGTHS are defined using 2 landmarks - the measurement is 
                found with distance between landmarks
       - CIRCUMFERENCES are defined with landmarks and joints - the 
                measurement is found by cutting the SMPL model with the 
                plane defined by a point (landmark point) and normal (
                vector connecting the two joints)
    3. If the body part is a CIRCUMFERENCE, a possible issue that arises is
       that the plane cutting results in multiple body part slices. To alleviate
       that, define the body part where the measurement should be located in 
       CIRCUMFERENCE_TO_BODYPARTS dict. This way, only slice in that body part is
       used for finding the measurement. The body parts are defined by the SMPL 
       face segmentation.
    '''
    
    # Dizionario delle misure di tipo LUNGHEZZA: ad ogni nome di misura è
    # associata una tupla di landmark (di solito 2) presi da
    # SMPL_LANDMARK_INDICES. La misura viene calcolata come distanza (semplice
    # o lungo la superficie) tra questi punti sulla mesh.
    LENGTHS = {"height":                       # altezza totale: testa -> talloni
                    (SMPL_LANDMARK_INDICES["HEAD_TOP"],
                     SMPL_LANDMARK_INDICES["HEELS"]
                     ),
               "shoulder to crotch height":    # altezza spalla -> inguine
                    (SMPL_LANDMARK_INDICES["SHOULDER_TOP"],
                     SMPL_LANDMARK_INDICES["INSEAM_POINT"]
                    ),
                "arm left length":             # lunghezza braccio sinistro: spalla -> polso
                    (SMPL_LANDMARK_INDICES["LEFT_SHOULDER"],
                     SMPL_LANDMARK_INDICES["LEFT_WRIST"]
                    ),
                "arm right length":
                    (SMPL_LANDMARK_INDICES["RIGHT_SHOULDER"], 
                     SMPL_LANDMARK_INDICES["RIGHT_WRIST"]
                    ),
                "inside leg height": 
                    (SMPL_LANDMARK_INDICES["LOW_LEFT_HIP"], 
                     SMPL_LANDMARK_INDICES["LEFT_ANKLE"]
                    ),
                "shoulder breadth": 
                    (SMPL_LANDMARK_INDICES["LEFT_SHOULDER"], 
                     SMPL_LANDMARK_INDICES["RIGHT_SHOULDER"]
                    ),
                # Voce "speciale": a differenza delle altre misure di lunghezza,
                # qui la coppia di landmark non usa i punti "generici"
                # (LEFT_SHOULDER/LEFT_ELBOW, lasciati come commento sotto per
                # memoria) ma i landmark in stile CAESAR (Rt. Acromion = punta
                # della spalla destra, Rt. Humeral Lateral Epicn = gomito
                # destro), ritenuti più precisi per isolare il tratto
                # spalla-gomito.
                "arm length (shoulder to elbow)":
                    (
                    #  SMPL_LANDMARK_INDICES["LEFT_SHOULDER"],
                    #  SMPL_LANDMARK_INDICES["LEFT_ELBOW"]
                    SMPL_LANDMARK_INDICES["Rt. Acromion"],
                    SMPL_LANDMARK_INDICES["Rt. Humeral Lateral Epicn"]
                    ),
                "crotch height":
                    (SMPL_LANDMARK_INDICES["CROTCH"],
                     SMPL_LANDMARK_INDICES["HEELS"]
                    ),
                "Hip circumference max height":
                    (SMPL_LANDMARK_INDICES["PUBIC_BONE"],
                     SMPL_LANDMARK_INDICES["HEELS"]
                    ),
                # Voce "speciale": a differenza di TUTTE le altre misure di
                # questo dizionario (definite con una tupla di 2 landmark),
                # questa è definita con una tupla di 4 landmark in sequenza
                # lungo il braccio (dalla base del collo fino al polso,
                # passando per spalla e gomito). Il commento FIXME sopra
                # segnala che la misura andrebbe idealmente calcolata come
                # distanza geodetica (lungo la superficie del corpo) sommando
                # i tratti tra questi 4 punti, invece di una singola distanza
                # in linea retta tra 2 soli punti come per le altre lunghezze.
                # FIXME: implement geodesic distance for this measurement
                "arm length (spine to wrist)":
                    (
                    #  SMPL_LANDMARK_INDICES["SHOULDER_TOP"],
                    #  SMPL_LANDMARK_INDICES["LEFT_WRIST"]
                        SMPL_LANDMARK_INDICES["Cervicale"],
                        SMPL_LANDMARK_INDICES["Rt. Acromion"],
                        SMPL_LANDMARK_INDICES["Rt. Humeral Lateral Epicn"],
                        SMPL_LANDMARK_INDICES["Rt. Ulnar Styloid"]
                    ),
               }

    # Dizionario delle misure di tipo CIRCONFERENZA. Ogni voce ha per valore
    # un dizionario con due chiavi:
    # - "LANDMARKS": lista di uno o più nomi di landmark (da
    #   SMPL_LANDMARK_INDICES) che indicano DOVE tagliare il corpo (il piano
    #   di taglio passa per quel punto, o per il punto medio se sono 2);
    # - "JOINTS": lista di ESATTAMENTE due nomi di giunto (da
    #   SMPL_IND2JOINT/JOINT2IND) il cui segmento definisce la NORMALE del
    #   piano di taglio, cioè la sua orientazione.
    # Tagliando la mesh con questo piano si ottiene una sezione, il cui
    # perimetro è la circonferenza misurata.
    CIRCUMFERENCES = {
        "head circumference":{"LANDMARKS":["HEAD_LEFT_TEMPLE"],   # taglio passante per la tempia sinistra
                               "JOINTS":["pelvis","spine3"]},       # normale = asse bacino-colonna (verticale)

        "neck circumference":{"LANDMARKS":["NECK_ADAM_APPLE"],    # taglio passante per il "pomo d'Adamo"
                               "JOINTS":["spine2","head"]},         # normale = asse colonna-testa

        "chest circumference":{"LANDMARKS":["LEFT_NIPPLE","RIGHT_NIPPLE"],  # taglio nel piano dei due capezzoli
                               "JOINTS":["pelvis","spine3"]},

        "waist circumference":{"LANDMARKS":["BELLY_BUTTON","BACK_BELLY_BUTTON"],
                               "JOINTS":["pelvis","spine3"]},

        "hip circumference":{"LANDMARKS":["PUBIC_BONE"],
                               "JOINTS":["pelvis","spine3"]},

        "wrist right circumference":{"LANDMARKS":["RIGHT_WRIST"],
                                    "JOINTS":["right_wrist","right_hand"]},
        
        "bicep right circumference":{"LANDMARKS":["RIGHT_BICEP"],
                                    "JOINTS":["right_shoulder","right_elbow"]},

        "forearm right circumference":{"LANDMARKS":["RIGHT_FOREARM"],
                                        "JOINTS":["right_elbow","right_wrist"]},
        
        "thigh left circumference":{"LANDMARKS":["LEFT_THIGH"],
                                    "JOINTS":["pelvis","spine3"]},
        
        "calf left circumference":{"LANDMARKS":["LEFT_CALF"],
                                    "JOINTS":["pelvis","spine3"]},

        "ankle left circumference":{"LANDMARKS":["LEFT_ANKLE"],
                                    "JOINTS":["pelvis","spine3"]},      
                    
                    }

    # Semplice lista con tutti i nomi di misura disponibili per SMPL, ottenuta
    # unendo le chiavi di LENGTHS e di CIRCUMFERENCES: serve per validare o
    # elencare le misure richiedibili al Measurer.
    possible_measurements = list(LENGTHS.keys()) + list(CIRCUMFERENCES.keys())

    # Dizionario che, per ogni misura di circonferenza, indica in quale/i
    # parte/i del corpo (secondo la segmentazione standard delle facce della
    # mesh SMPL) va cercata la sezione giusta. Serve perché il piano di taglio
    # definito in CIRCUMFERENCES può intersecare la mesh in più punti (es.
    # anche le braccia, oltre al busto): restringendo la ricerca a queste
    # parti si evita di scegliere per errore la sezione sbagliata. Il valore
    # può essere una singola stringa (una sola parte del corpo) oppure una
    # lista di stringhe (più parti, quando la sezione attraversa più zone
    # della segmentazione).
    CIRCUMFERENCE_TO_BODYPARTS = {
        "head circumference": "head",                  # una sola parte del corpo
        "neck circumference":"neck",
        "chest circumference":["spine1","spine2"],      # due parti: il taglio attraversa entrambe
        "waist circumference":["hips","spine"],
        "hip circumference":"hips",
        "wrist right circumference":["rightHand","rightForeArm"],
        "bicep right circumference":"rightArm",
        "forearm right circumference":"rightForeArm",
        "thigh left circumference": "leftUpLeg",
        "calf left circumference": "leftLeg",
        "ankle left circumference": "leftLeg",
    }



class SMPLXMeasurementDefinitions():
    '''
    Definition of SMPLX measurements.

    To add a new measurement:
    1. add it to the measurement_types dict and set the type:
       LENGTH or CIRCUMFERENCE
    2. depending on the type, define the measurement in LENGTHS or 
       CIRCUMFERENCES dict
       - LENGTHS are defined using 2 landmarks - the measurement is 
                found with distance between landmarks
       - CIRCUMFERENCES are defined with landmarks and joints - the 
                measurement is found by cutting the SMPLX model with the 
                plane defined by a point (landmark point) and normal (
                vector connecting the two joints)
    3. If the body part is a CIRCUMFERENCE, a possible issue that arises is
       that the plane cutting results in multiple body part slices. To alleviate
       that, define the body part where the measurement should be located in 
       CIRCUMFERENCE_TO_BODYPARTS dict. This way, only slice in that body part is
       used for finding the measurement. The body parts are defined by the SMPL 
       face segmentation.
    '''
    
    # Stesso ruolo di SMPLMeasurementDefinitions.LENGTHS, ma con gli indici
    # di SMPLX_LANDMARK_INDICES. Rispetto alla versione SMPL, qui mancano
    # "arm length (shoulder to elbow)", "crotch height",
    # "Hip circumference max height" e "arm length (spine to wrist)", perché
    # SMPLX non definisce i landmark in stile CAESAR usati per quelle misure.
    LENGTHS = {"height":                       # altezza totale: testa -> talloni
                    (SMPLX_LANDMARK_INDICES["HEAD_TOP"],
                     SMPLX_LANDMARK_INDICES["HEELS"]
                     ),
               "shoulder to crotch height":    # altezza spalla -> inguine
                    (SMPLX_LANDMARK_INDICES["SHOULDER_TOP"],
                     SMPLX_LANDMARK_INDICES["INSEAM_POINT"]
                    ),
                "arm left length":             # lunghezza braccio sinistro: spalla -> polso
                    (SMPLX_LANDMARK_INDICES["LEFT_SHOULDER"],
                     SMPLX_LANDMARK_INDICES["LEFT_WRIST"]
                    ),
                "arm right length":
                    (SMPLX_LANDMARK_INDICES["RIGHT_SHOULDER"], 
                     SMPLX_LANDMARK_INDICES["RIGHT_WRIST"]
                    ),
                "inside leg height": 
                    (SMPLX_LANDMARK_INDICES["LOW_LEFT_HIP"], 
                     SMPLX_LANDMARK_INDICES["LEFT_ANKLE"]
                    ),
                "shoulder breadth": 
                    (SMPLX_LANDMARK_INDICES["LEFT_SHOULDER"], 
                     SMPLX_LANDMARK_INDICES["RIGHT_SHOULDER"]
                    ),
               }

    # Stesso ruolo di SMPLMeasurementDefinitions.CIRCUMFERENCES (vedi il
    # commento lì per il significato di "LANDMARKS"/"JOINTS"), qui con gli
    # indici e i nomi di giunto validi per SMPLX. Alcune voci usano giunti
    # diversi rispetto a SMPL (segnalato inline dove succede).
    CIRCUMFERENCES = {
        "head circumference":{"LANDMARKS":["HEAD_LEFT_TEMPLE"],   # taglio passante per la tempia sinistra
                               "JOINTS":["pelvis","spine3"]},       # normale = asse bacino-colonna

        "neck circumference":{"LANDMARKS":["NECK_ADAM_APPLE"],    # taglio passante per il "pomo d'Adamo"
                               "JOINTS":["spine1","spine3"]},       # diverso da SMPL (lì: spine2, head)

        "chest circumference":{"LANDMARKS":["LEFT_NIPPLE","RIGHT_NIPPLE"],  # taglio nel piano dei due capezzoli
                               "JOINTS":["pelvis","spine3"]},

        "waist circumference":{"LANDMARKS":["BELLY_BUTTON","BACK_BELLY_BUTTON"],
                               "JOINTS":["pelvis","spine3"]},
        
        "hip circumference":{"LANDMARKS":["PUBIC_BONE"],
                               "JOINTS":["pelvis","spine3"]},
        
        "wrist right circumference":{"LANDMARKS":["RIGHT_WRIST"],
                                    "JOINTS":["right_wrist","right_elbow"]}, # different from SMPL
        
        "bicep right circumference":{"LANDMARKS":["RIGHT_BICEP"],
                                    "JOINTS":["right_shoulder","right_elbow"]},

        "forearm right circumference":{"LANDMARKS":["RIGHT_FOREARM"],
                                        "JOINTS":["right_elbow","right_wrist"]},
        
        "thigh left circumference":{"LANDMARKS":["LEFT_THIGH"],
                                    "JOINTS":["pelvis","spine3"]},
        
        "calf left circumference":{"LANDMARKS":["LEFT_CALF"],
                                    "JOINTS":["pelvis","spine3"]},

        "ankle left circumference":{"LANDMARKS":["LEFT_ANKLE"],
                                    "JOINTS":["pelvis","spine3"]},      
                    
                    }
    
    # Stesso ruolo di possible_measurements/CIRCUMFERENCE_TO_BODYPARTS della
    # classe SMPLMeasurementDefinitions (vedi i commenti lì): elenco di tutte
    # le misure disponibili per SMPLX, e per ciascuna circonferenza la/le
    # parte/i del corpo (segmentazione SMPL) in cui cercare la sezione giusta
    # (stringa singola o lista di stringhe).
    possible_measurements = list(LENGTHS.keys()) + list(CIRCUMFERENCES.keys())

    CIRCUMFERENCE_TO_BODYPARTS = {
        "head circumference": "head",
        "neck circumference":"neck",
        "chest circumference":["spine1","spine2"],
        "waist circumference":["hips","spine"],
        "hip circumference":"hips",
        "wrist right circumference":["rightHand","rightForeArm"],
        "bicep right circumference":"rightArm",
        "forearm right circumference":"rightForeArm",
        "thigh left circumference": "leftUpLeg",
        "calf left circumference": "leftLeg",
        "ankle left circumference": "leftLeg",
    }