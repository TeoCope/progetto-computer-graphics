import smplx
import torch
import os

# Questo file definisce i "giunti" (joints) dello scheletro usato dai modelli
# di corpo SMPL/SMPLX: lo scheletro è la catena di ossa/articolazioni virtuali
# che permette di posare (ruotare braccia, gambe, ecc.) la mesh. Qui non si
# definiscono le posizioni 3D dei giunti, ma solo la loro numerazione e i
# nomi, così il resto del codice può riferirsi a un giunto per indice
# (come richiesto dal modello) o per nome (più leggibile per un umano).
#from https://meshcapade.wiki/SMPL

# Numero totale di giunti dello scheletro SMPL (corpo "base", senza mani/viso).
SMPL_NUM_JOINTS = 24

# Dizionario indice -> nome del giunto per SMPL. L'indice è la posizione del
# giunto nell'array di rotazioni (pose) e nell'output del modello SMPL;
# il nome è quello standard usato nella letteratura/documentazione SMPL.
# Esempio di lettura: il giunto 0 è il "pelvis" (bacino, radice dello
# scheletro), il giunto 1 è l'anca sinistra, il giunto 4 il ginocchio
# sinistro, e così via risalendo dal basso verso l'alto lungo il corpo.
SMPL_IND2JOINT = {
    0: 'pelvis',
     1: 'left_hip',
     2: 'right_hip',
     3: 'spine1',
     4: 'left_knee',
     5: 'right_knee',
     6: 'spine2',
     7: 'left_ankle',
     8: 'right_ankle',
     9: 'spine3',
    10: 'left_foot',
    11: 'right_foot',
    12: 'neck',
    13: 'left_collar',
    14: 'right_collar',
    15: 'head',
    16: 'left_shoulder',
    17: 'right_shoulder',
    18: 'left_elbow',
    19: 'right_elbow',
    20: 'left_wrist',
    21: 'right_wrist',
    22: 'left_hand',
    23: 'right_hand'
}

# Dizionario "inverso" nome -> indice, generato automaticamente a partire da
# SMPL_IND2JOINT (basta scambiare chiave e valore). Utile quando nel codice si
# conosce il nome del giunto (es. "left_knee") e serve il suo indice numerico
# per indicizzare array/tensori del modello.
SMPL_JOINT2IND = {name:ind for ind,name in SMPL_IND2JOINT.items()}




# Numero totale di giunti dello scheletro SMPLX: rispetto a SMPL (24) include
# anche i giunti di mani e viso, quindi sono molti di più (55).
SMPLX_NUM_JOINTS = 55

# Dizionario indice -> nome del giunto per SMPLX, stesso ruolo di
# SMPL_IND2JOINT. I primi 22 giunti (indici 0-21) coincidono concettualmente
# con quelli "di base" di SMPL (bacino, anche, ginocchia, colonna, spalle,
# gomiti, polsi); da qui in poi la lista prosegue con giunti che SMPL non ha:
# mascella/occhi (jaw, left_eye, right_eye) e poi le tre falangi di ciascun
# dito di entrambe le mani (index/middle/pinky/ring/thumb, 1/2/3), perché
# SMPLX modella anche mani e viso in dettaglio.
SMPLX_IND2JOINT = {
    0: 'pelvis',
     1: 'left_hip',
     2: 'right_hip',
     3: 'spine1',
     4: 'left_knee',
     5: 'right_knee',
     6: 'spine2',
     7: 'left_ankle',
     8: 'right_ankle',
     9: 'spine3',
    10: 'left_foot',
    11: 'right_foot',
    12: 'neck',
    13: 'left_collar',
    14: 'right_collar',
    15: 'head',
    16: 'left_shoulder',
    17: 'right_shoulder',
    18: 'left_elbow',
    19: 'right_elbow',
    20: 'left_wrist',
    21: 'right_wrist',
    22: 'jaw',
    23: 'left_eye',
    24: 'right_eye',
    25: 'left_index1',
    26: 'left_index2',
    27: 'left_index3',
    28: 'left_middle1',
    29: 'left_middle2',
    30: 'left_middle3',
    31: 'left_pinky1',
    32: 'left_pinky2',
    33: 'left_pinky3',
    34: 'left_ring1',
    35: 'left_ring2',
    36: 'left_ring3',
    37: 'left_thumb1',
    38: 'left_thumb2',
    39: 'left_thumb3',
    40: 'right_index1',
    41: 'right_index2',
    42: 'right_index3',
    43: 'right_middle1',
    44: 'right_middle2',
    45: 'right_middle3',
    46: 'right_pinky1',
    47: 'right_pinky2',
    48: 'right_pinky3',
    49: 'right_ring1',
    50: 'right_ring2',
    51: 'right_ring3',
    52: 'right_thumb1',
    53: 'right_thumb2',
    54: 'right_thumb3'
}

# Dizionario inverso nome -> indice per SMPLX, stesso principio di
# SMPL_JOINT2IND.
SMPLX_JOINT2IND = {name:ind for ind,name in SMPLX_IND2JOINT.items()}

# Funzione (non un dizionario di dati, ma un piccolo helper): carica il
# modello di corpo SMPL/SMPLX da disco tramite la libreria "smplx" e ne
# restituisce lo "J_regressor", cioè la matrice che permette di calcolare
# la posizione 3D dei giunti a partire dai vertici della mesh. Viene usata
# altrove nel progetto (es. dal Measurer) per sapere dove si trovano i
# giunti definiti sopra (pelvis, spine3, ecc.) su un corpo specifico.
def get_joint_regressor(body_model_type, body_model_root, gender="MALE", num_thetas=24):
    '''
    Extract joint regressor from SMPL body model
    :param body_model_type: str of body model type (smpl or smplx, etc.)
    :param body_model_root: str of location of folders where smpl/smplx 
                            inside which .pkl models 
    
    Return:
    :param model.J_regressor: torch.tensor (23,N) used to 
                              multiply with body model to get 
                              joint locations
    '''

    model = smplx.create(model_path=body_model_root, 
                        model_type=body_model_type,
                        gender=gender, 
                        use_face_contour=False,
                        num_betas=10,
                        body_pose=torch.zeros((1, num_thetas-1 * 3)),
                        ext='pkl')
    return model.J_regressor