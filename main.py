# This script is modified from https://github.com/isl-org/Open3D/blob/master/examples/python/gui/vis-gui.py

# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2018-2021 www.open3d.org
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
# ----------------------------------------------------------------------------

# ============================================================================
# SEZIONE: IMPORT DELLE LIBRERIE
# Qui importiamo tutti i moduli/librerie che servono al programma.
# In Python, "import" carica del codice scritto da altri (o da noi in altri
# file) per poterlo riusare, senza doverlo riscrivere da zero.
# ============================================================================
import os  # Funzioni per interagire col sistema operativo (percorsi di file, cartelle, ecc.)
import sys  # Funzioni legate all'interprete Python (es. argomenti da riga di comando, path di ricerca moduli)
import copy  # Serve per fare copie "profonde" (deepcopy) di oggetti, così modificarne una non tocca l'originale
import glob  # Permette di cercare file usando pattern con caratteri jolly (es. "*.obj")
import torch  # Libreria di deep learning: qui viene usata per i tensori numerici richiesti dai modelli SMPL/SMPLX/MANO/FLAME
import joblib  # Libreria per salvare/caricare oggetti Python su disco (es. i parametri della posa/forma del corpo)
import platform  # Permette di scoprire su quale sistema operativo gira il programma (Windows, macOS, Linux)
import argparse  # Libreria standard per leggere argomenti passati da riga di comando (es. "python main.py --model smpl")
import numpy as np  # Libreria per calcoli numerici e vettoriali (array, matrici); "np" è l'alias convenzionale
import open3d as o3d  # Libreria principale per creare mesh 3D, gestire la finestra di visualizzazione e il rendering
from loguru import logger  # Libreria di logging: serve per stampare messaggi di debug/informazione in modo ordinato
import open3d.visualization.gui as gui  # Sotto-modulo di Open3D con i "widget" (bottoni, slider, finestre) per costruire l'interfaccia grafica
import scipy.spatial.transform.rotation as R  # Modulo di SciPy per convertire rotazioni tra rappresentazioni diverse (es. angoli di Eulero <-> axis-angle)
import open3d.visualization.rendering as rendering  # Sotto-modulo di Open3D per gestire materiali, luci e la scena 3D vera e propria


# Import di alcune funzioni/costanti definite nel file utils.py del progetto:
# nomi dei giunti (joint) dei vari modelli di corpo, e una funzione per creare
# un piano a scacchiera (usato come "pavimento" nella scena 3D).
from utils import (
    get_checkerboard_plane,
    load_obj_triangle_uvs,
    smpl_joint_names,
    smplx_body_joint_names,
    hand_joint_names,
    LEFT_HAND_KEYPOINT_NAMES,
    RIGHT_HAND_KEYPOINT_NAMES,
    HEAD_KEYPOINT_NAMES,
    FLAME_KEYPOINT_NAMES,
    FOOT_KEYPOINT_NAMES,
    SMPL_NAMES,
    SMPLX_NAMES,
    MANO_NAMES,
)
from simple_ik import simple_ik_solver  # Import del nostro semplice risolutore di IK (Inverse Kinematics): calcola gli angoli delle articolazioni per far raggiungere a una mano/piede un punto target

# Add submodule path
# Aggiungiamo al "sys.path" (l'elenco di cartelle in cui Python cerca i moduli)
# la cartella del sottoprogetto "SMPL-Anthropometry-master", così possiamo
# importare da lì il modulo "measure" anche se si trova in una sottocartella.
sys.path.append(os.path.join(os.path.dirname(__file__), "SMPL-Anthropometry-master"))
from measure import MeasureBody  # Classe che calcola le misure antropometriche (altezza, circonferenze, ecc.) a partire dalla mesh del corpo

import json  # Per salvare/leggere i parametri dell'avatar e i preset in formato JSON (interscambiabile con Unity)
# Logica di fitting condivisa con lo script offline build_presets.py:
# stessa funzione usata sia dal bottone "Fit Avatar to Targets" della GUI
# sia per precalcolare le betas dei preset di avatar.
from fitting import fit_betas_to_measurements, load_presets

# Verifichiamo se il sistema operativo è macOS: alcune impostazioni della GUI
# (es. posizione del menu) si comportano diversamente su Mac rispetto a Windows/Linux.
isMacOS = (platform.system() == "Darwin")


# ============================================================================
# CLASSE Settings
# Questa classe NON rappresenta un oggetto della scena 3D, ma raccoglie tutte
# le "impostazioni" correnti di rendering: che materiale/shader usare, come
# sono posizionate le luci, i profili di illuminazione predefiniti, ecc.
# Tenere queste impostazioni in una classe separata (invece che sparse in
# variabili globali) rende il codice più ordinato e facile da passare in giro.
# ============================================================================
class Settings:
    # Nomi degli "shader" (programmi che decidono come colorare i pixel della mesh)
    # disponibili in Open3D. Sono definiti come costanti di classe per evitare
    # di scrivere le stesse stringhe a mano in più punti del codice (e sbagliarle).
    UNLIT = "defaultUnlit"  # Materiale "piatto": nessun effetto di luce, colori uniformi
    LIT = "defaultLit"  # Materiale realistico che reagisce alla luce (PBR - Physically Based Rendering)
    NORMALS = "normals"  # Shader di debug: colora la mesh in base alla direzione delle normali
    DEPTH = "depth"  # Shader di debug: colora la mesh in base alla distanza dalla camera

    # Nomi (stringhe) usati nei menu a tendina per scegliere un "profilo di illuminazione"
    DEFAULT_PROFILE_NAME = "Bright day with sun at +Y [default]"
    POINT_CLOUD_PROFILE_NAME = "Cloudy day (no direct sun)"
    CUSTOM_PROFILE_NAME = "Custom"
    # Dizionario dei profili di illuminazione predefiniti: per ogni nome scelto
    # dall'utente nel menu, associamo un insieme di parametri (intensità della
    # luce ambientale IBL, intensità del sole, direzione del sole, ecc.) che
    # verranno applicati tutti insieme alla scena.
    LIGHTING_PROFILES = {
        DEFAULT_PROFILE_NAME: {
            "ibl_intensity": 45000,
            "sun_intensity": 45000,
            "sun_dir": [0.577, -0.577, -0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "Bright day with sun at -Y": {
            "ibl_intensity": 45000,
            "sun_intensity": 45000,
            "sun_dir": [0.577, 0.577, 0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "Bright day with sun at +Z": {
            "ibl_intensity": 45000,
            "sun_intensity": 45000,
            "sun_dir": [0.577, 0.577, -0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "Less Bright day with sun at +Y": {
            "ibl_intensity": 35000,
            "sun_intensity": 50000,
            "sun_dir": [0.577, -0.577, -0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "Less Bright day with sun at -Y": {
            "ibl_intensity": 35000,
            "sun_intensity": 50000,
            "sun_dir": [0.577, 0.577, 0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "Less Bright day with sun at +Z": {
            "ibl_intensity": 35000,
            "sun_intensity": 50000,
            "sun_dir": [0.577, 0.577, -0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        # Gli altri profili qui sotto seguono lo stesso schema del primo:
        # cambia solo "ibl_intensity", "sun_intensity" e "sun_dir" (la
        # direzione da cui arriva la luce del sole), quindi non li
        # ricommentiamo uno per uno.
        POINT_CLOUD_PROFILE_NAME: {
            "ibl_intensity": 60000,
            "sun_intensity": 50000,
            "use_ibl": True,
            "use_sun": False,
            # "ibl_rotation":
        },
    }

    # Nome del materiale scelto di default per la mesh
    DEFAULT_MATERIAL_NAME = "Polished ceramic [default]"
    # Dizionario dei "materiali predefiniti" (prefab): ogni voce imposta le
    # proprietà fisiche del materiale PBR (quanto è metallico, ruvido,
    # riflettente, con quanta "vernice trasparente"/clearcoat, ecc.).
    # Questi valori vengono applicati al materiale corrente quando l'utente
    # sceglie una voce dal menu a tendina dei materiali.
    PREFAB = {
        DEFAULT_MATERIAL_NAME: {
            "metallic": 0.0,
            "roughness": 0.7,
            "reflectance": 0.5,
            "clearcoat": 0.2,
            "clearcoat_roughness": 0.2,
            "anisotropy": 0.0
        },
        # Le voci successive seguono lo stesso schema di "DEFAULT_MATERIAL_NAME":
        # cambiano solo i valori numerici delle proprietà del materiale.
        "Metal (rougher)": {
            "metallic": 1.0,
            "roughness": 0.5,
            "reflectance": 0.9,
            "clearcoat": 0.0,
            "clearcoat_roughness": 0.0,
            "anisotropy": 0.0
        },
        "Metal (smoother)": {
            "metallic": 1.0,
            "roughness": 0.3,
            "reflectance": 0.9,
            "clearcoat": 0.0,
            "clearcoat_roughness": 0.0,
            "anisotropy": 0.0
        },
        "Plastic": {
            "metallic": 0.0,
            "roughness": 0.5,
            "reflectance": 0.5,
            "clearcoat": 0.5,
            "clearcoat_roughness": 0.2,
            "anisotropy": 0.0
        },
        "Glazed ceramic": {
            "metallic": 0.0,
            "roughness": 0.5,
            "reflectance": 0.9,
            "clearcoat": 1.0,
            "clearcoat_roughness": 0.1,
            "anisotropy": 0.0
        },
        "Clay": {
            "metallic": 0.0,
            "roughness": 1.0,
            "reflectance": 0.5,
            "clearcoat": 0.1,
            "clearcoat_roughness": 0.287,
            "anisotropy": 0.0
        },
    }

    def __init__(self):
        # Costruttore: qui definiamo i valori iniziali (di default) di tutte
        # le impostazioni della scena e del rendering.
        self.mouse_model = gui.SceneWidget.Controls.ROTATE_CAMERA  # Modalità mouse di default: ruotare la camera intorno alla scena
        self.bg_color = gui.Color(1, 1, 1)  # Colore di sfondo della finestra 3D (bianco, valori RGB da 0 a 1)
        self.show_skybox = False  # Se mostrare o no lo sfondo "skybox" (cielo/ambiente)
        self.show_axes = True  # Se mostrare gli assi X/Y/Z di riferimento
        self.show_ground = True  # Se mostrare il piano/pavimento a scacchiera
        self.use_ibl = True  # IBL = Image-Based Lighting, illuminazione ambientale basata su un'immagine panoramica
        self.use_sun = True  # Se usare anche una luce direzionale che simula il sole
        self.new_ibl_name = None  # clear to None after loading
        self.ibl_intensity = 45000  # Intensità della luce ambientale (IBL)
        self.sun_intensity = 45000  # Intensità della luce solare
        self.sun_dir = [0.577, -0.577, -0.577]  # Direzione da cui arriva la luce del sole (vettore 3D)
        self.sun_color = gui.Color(1, 1, 1)  # Colore della luce solare (bianco)

        self.apply_material = True  # clear to False after processing
        # Creiamo un materiale distinto per ciascuno shader disponibile: in
        # questo modo, cambiando shader, non perdiamo le proprietà impostate
        # sugli altri materiali (rimangono "in memoria" pronti per essere
        # riusati se l'utente torna a selezionarli).
        self._materials = {
            Settings.LIT: rendering.MaterialRecord(),
            Settings.UNLIT: rendering.MaterialRecord(),
            Settings.NORMALS: rendering.MaterialRecord(),
            Settings.DEPTH: rendering.MaterialRecord()
        }
        # Impostiamo un colore di base grigio chiaro e lo shader corretto per
        # i materiali "lit" (con luce) e "unlit" (senza luce); gli shader di
        # debug (normali/profondità) non hanno bisogno di un colore di base.
        self._materials[Settings.LIT].base_color = [0.9, 0.9, 0.9, 1.0]
        self._materials[Settings.LIT].shader = Settings.LIT
        self._materials[Settings.UNLIT].base_color = [0.9, 0.9, 0.9, 1.0]
        self._materials[Settings.UNLIT].shader = Settings.UNLIT
        self._materials[Settings.NORMALS].shader = Settings.NORMALS
        self._materials[Settings.DEPTH].shader = Settings.DEPTH

        # Conveniently, assigning from self._materials[...] assigns a reference,
        # not a copy, so if we change the property of a material, then switch
        # to another one, then come back, the old setting will still be there.
        # (Commento originale: in Python assegnare un dizionario/oggetto a una
        # variabile copia solo il "riferimento", non i dati: per questo motivo
        # modificare self.material modifica anche l'oggetto dentro _materials.)
        self.material = self._materials[Settings.LIT]  # Materiale attivo di default: quello "lit" (con illuminazione realistica)

    def set_material(self, name):
        # Cambia il materiale attivo scegliendolo per nome tra quelli disponibili
        # e segnala (tramite apply_material) che va riapplicato alla mesh.
        self.material = self._materials[name]
        self.apply_material = True

    def apply_material_prefab(self, name):
        # Applica un "prefab" di materiale (uno dei set di valori in PREFAB)
        # al materiale corrente. Funziona solo con lo shader LIT, perché solo
        # quello ha proprietà fisiche come metallic/roughness/reflectance.
        assert (self.material.shader == Settings.LIT)
        prefab = Settings.PREFAB[name]
        # Per ogni proprietà nel prefab (es. "metallic"), impostiamo l'attributo
        # corrispondente sul materiale con il prefisso "base_" (es. "base_metallic").
        # "setattr" permette di impostare un attributo di un oggetto conoscendone
        # solo il nome come stringa, utile quando i nomi non sono fissi a priori.
        for key, val in prefab.items():
            setattr(self.material, "base_" + key, val)

    def apply_lighting_profile(self, name):
        # Applica un intero profilo di illuminazione (uno dei LIGHTING_PROFILES)
        # copiando ogni suo valore direttamente come attributo di questa istanza
        # di Settings (es. self.ibl_intensity, self.sun_dir, ecc.).
        profile = Settings.LIGHTING_PROFILES[name]
        for key, val in profile.items():
            setattr(self, key, val)


# ============================================================================
# CLASSE AppWindow
# Questa è la classe "principale" del programma: rappresenta l'intera finestra
# dell'applicazione. Contiene sia i dati (la mesh corrente, i parametri del
# modello di corpo, le impostazioni) sia tutti i metodi "callback" che
# vengono chiamati quando l'utente interagisce con la GUI (click su bottoni,
# spostamento di slider, mouse sulla scena 3D, ecc.).
# ============================================================================
class AppWindow:
    # Identificatori numerici delle voci di menu (File > Open, Export, ecc.).
    # Vengono usati come "id" quando si registra la funzione da chiamare al click.
    MENU_OPEN = 1
    MENU_EXPORT = 2
    MENU_QUIT = 3
    MENU_SAVE = 4
    MENU_EXPORT_OBJ = 5
    MENU_SHOW_SETTINGS = 11
    MENU_ABOUT = 21

    DEFAULT_IBL = "default"  # Nome della mappa di illuminazione ambientale (Image-Based Lighting) di default

    # Nomi mostrati nel menu a tendina dei materiali/shader, e gli shader
    # Open3D corrispondenti (stesso ordine, indice per indice).
    MATERIAL_NAMES = ["Lit", "Unlit", "Normals", "Depth"]
    MATERIAL_SHADERS = [
        Settings.LIT, Settings.UNLIT, Settings.NORMALS, Settings.DEPTH
    ]

    # Elenco dei modelli di corpo 3D supportati dal programma.
    BODY_MODEL_NAMES = ["SMPL", "SMPLX", "MANO", "FLAME"]
    # Per ogni modello, i "generi" disponibili (alcuni modelli, come MANO,
    # hanno solo la versione neutra).
    BODY_MODEL_GENDERS = {
        'SMPL': ['neutral', 'male', 'female'],
        'SMPLX': ['neutral', 'male', 'female'],
        'MANO': ['neutral'],
        'FLAME': ['neutral', 'male', 'female']
    }
    # Etichette leggibili (in italiano) da mostrare accanto agli slider dei
    # "betas": i betas sono i parametri di forma del corpo (i primi controllano
    # aspetti "grossolani" come altezza/peso, gli altri dettagli più fini).
    BETA_LABELS = {
        'SMPL': ['Stazza/Altezza', 'Peso/Larghezza', 'Proporzioni', 'Tronco/Gambe', 'Spalle', 'Dettaglio 6', 'Dettaglio 7', 'Dettaglio 8', 'Dettaglio 9', 'Dettaglio 10'],
        'SMPLX': ['Stazza/Altezza', 'Peso/Larghezza', 'Proporzioni', 'Tronco/Gambe', 'Spalle', 'Dettaglio 6', 'Dettaglio 7', 'Dettaglio 8', 'Dettaglio 9', 'Dettaglio 10'],
        'MANO': ['Dim. Mano', 'Spessore Dita', 'Dettaglio 3', 'Dettaglio 4', 'Dettaglio 5', 'Dettaglio 6', 'Dettaglio 7', 'Dettaglio 8', 'Dettaglio 9', 'Dettaglio 10'],
        'FLAME': ['Dim. Testa', 'Rotondità Viso', 'Dettaglio 3', 'Dettaglio 4', 'Dettaglio 5', 'Dettaglio 6', 'Dettaglio 7', 'Dettaglio 8', 'Dettaglio 9', 'Dettaglio 10'],
    }
    # Numero di parametri "beta" (di forma) per ciascun modello: qui sono
    # tutti 10, ma tenerlo come dizionario permette di cambiarlo facilmente
    # in futuro per un singolo modello senza toccare il resto del codice.
    BODY_MODEL_N_BETAS = {
        'SMPL': 10,
        'SMPLX': 10,
        'MANO': 10,
        'FLAME': 10,
    }
    CAM_FIRST = True  # Flag: indica se la camera deve essere posizionata automaticamente al primo caricamento della mesh

    PRELOADED_BODY_MODELS = {}  # Cache dei modelli di corpo già caricati da disco, per non doverli ricaricare ogni volta (operazione lenta)

    # Parametri di posa di default (tutti a zero = posa "a riposo"/T-pose) per
    # ciascun modello. torch.zeros(1, N, 3) crea un tensore di N angoli (uno
    # per giunto), ciascuno rappresentato da 3 valori (rotazione in axis-angle),
    # con una dimensione iniziale "1" che rappresenta il batch (qui: un solo corpo).
    POSE_PARAMS = {
        'SMPL': {
            'body_pose': torch.zeros(1, 23, 3),
            'global_orient': torch.zeros(1, 1, 3),
        },
        'SMPLX': {
            'body_pose': torch.zeros(1, 21, 3),
            'global_orient': torch.zeros(1, 1, 3),
            'left_hand_pose': torch.zeros(1, 15, 3),
            'right_hand_pose': torch.zeros(1, 15, 3),
            'jaw_pose': torch.zeros(1, 1, 3),
            'leye_pose': torch.zeros(1, 1, 3),
            'reye_pose': torch.zeros(1, 1, 3),
        },
        'MANO': {
            'hand_pose': torch.zeros(1, 15, 3),
            'global_orient': torch.zeros(1, 1, 3),
        },
        'FLAME': {
            'global_orient': torch.zeros(1, 1, 3),
            'jaw_pose': torch.zeros(1, 1, 3),
            'neck_pose': torch.zeros(1, 1, 3),
            'leye_pose': torch.zeros(1, 1, 3),
            'reye_pose': torch.zeros(1, 1, 3),
        },
    }

    # Per ciascun modello, associa ad ogni gruppo di parametri di posa
    # (es. 'body_pose') la lista dei nomi dei giunti corrispondenti, nello
    # stesso ordine in cui compaiono nel tensore della posa. Serve per poter
    # mostrare nella GUI il nome del giunto invece di un semplice indice numerico.
    JOINT_NAMES = {
        'SMPL': {
            'global_orient': ['root'],
            'body_pose': smpl_joint_names,
        },
        'SMPLX': {
            'global_orient': ['root'],
            'body_pose': smplx_body_joint_names,
            'left_hand_pose': hand_joint_names,
            'right_hand_pose': hand_joint_names,
            'jaw_pose': ['jaw'],
            'leye_pose': ['leye'],
            'reye_pose': ['reye'],
        },
        'MANO': {
            'global_orient': ['root'],
            'hand_pose': hand_joint_names,
        },
        'FLAME': {
            'global_orient': ['root'],
            'jaw_pose': ['jaw'],
            'neck_pose': ['neck'],
            'leye_pose': ['leye'],
            'reye_pose': ['reye'],
        },
    }

    # Nomi dei "keypoint" (punti notevoli, es. articolazioni della mano o del
    # viso) usati per ciascun modello, importati da utils.py.
    KEYPOINT_NAMES = {
        'SMPL': SMPL_NAMES,
        'SMPLX': SMPLX_NAMES,
        'MANO': MANO_NAMES,
        'FLAME': FLAME_KEYPOINT_NAMES,
    }

    # Variabili di classe usate come stato "globale" condiviso: la lista dei
    # giunti correnti, il giunto attualmente selezionato dall'utente (per IK)
    # e la traslazione applicata al corpo. Iniziano a None e vengono valorizzate
    # più avanti quando si carica un modello.
    JOINTS = None
    SELECTED_JOINT = None
    BODY_TRANSL = None

    # Cartella con gli asset per la texture dell'avatar: il template UV
    # ufficiale "smpl_uv.obj" (da scaricare da smpl.is.tue.mpg.de) e una o
    # più immagini texture (.png/.jpg). Vedi data/textures/README.md.
    TEXTURE_DIR = 'data/textures'
    UV_TEMPLATE_FILE = 'smpl_uv.obj'

    def __init__(self, width, height):
        # Costruttore della finestra principale: qui viene creata TUTTA la
        # GUI (scena 3D + pannello impostazioni + menu). Il codice è lungo
        # perché costruisce ogni singolo controllo (bottone, slider, ecc.)
        # a mano, ma segue sempre lo stesso pattern: crea il widget, imposta
        # le sue proprietà, collega una funzione di callback, poi lo aggiunge
        # a un layout (contenitore) che lo posiziona nella finestra.
        self.settings = Settings()  # Le impostazioni di rendering correnti (vedi classe Settings sopra)
        resource_path = gui.Application.instance.resource_path
        self.settings.new_ibl_name = resource_path + "/" + AppWindow.DEFAULT_IBL

        # Creiamo la finestra principale dell'applicazione con titolo "Open3D"
        # e le dimensioni (larghezza, altezza) passate come parametri.
        self.window = gui.Application.instance.create_window(
            "Open3D", width, height)
        w = self.window  # to make the code more concise

        # 3D widget
        # Il widget che mostra la scena 3D vera e propria (la mesh, gli assi,
        # il pavimento, ecc.) e gestisce l'interazione con il mouse.
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(w.renderer)  # La "scena" Open3D collegata al renderer della finestra
        self._scene.set_on_sun_direction_changed(self._on_sun_dir)  # Callback chiamata quando l'utente ruota la direzione del sole con il mouse

        # ---- Settings panel ----
        # Rather than specifying sizes in pixels, which may vary in size based
        # on the monitor, especially on macOS which has 220 dpi monitors, use
        # the em-size. This way sizings will be proportional to the font size,
        # which will create a more visually consistent size across platforms.
        # (In pratica: "em" è la dimensione del font corrente, usata come
        # unità di misura per margini e spaziature, così la GUI resta
        # proporzionata su schermi con densità di pixel diverse.)
        em = w.theme.font_size
        self.em = em
        separation_height = int(round(0.5 * em))

        # Widgets are laid out in layouts: gui.Horiz, gui.Vert,
        # gui.CollapsableVert, and gui.VGrid. By nesting the layouts we can
        # achieve complex designs. Usually we use a vertical layout as the
        # topmost widget, since widgets tend to be organized from top to bottom.
        # Within that, we usually have a series of horizontal layouts for each
        # row. All layouts take a spacing parameter, which is the spacing
        # between items in the widget, and a margins parameter, which specifies
        # the spacing of the left, top, right, bottom margins. (This acts like
        # the 'padding' property in CSS.)
        # (In pratica: gui.Vert = layout verticale, gui.Horiz = layout
        # orizzontale, gui.CollapsableVert = sezione richiudibile,
        # gui.VGrid = griglia. Si "annidano" layout dentro altri layout
        # per costruire l'interfaccia, un po' come le <div> annidate in HTML.)
        self._settings_panel = gui.Vert(
            0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))

        # Create a collapsable vertical widget, which takes up enough vertical
        # space for all its children when open, but only enough for text when
        # closed. This is useful for property pages, so the user can hide sets
        # of properties they rarely use.
        # Sezione richiudibile "View controls" (controlli di visualizzazione:
        # modalità del mouse per interagire con la scena 3D).
        view_ctrls = gui.CollapsableVert("View controls", 0.25 * em,
                                         gui.Margins(em, 0, 0, 0))

        view_ctrls.set_is_open(False)  # Parte chiusa/collassata di default
        # Creiamo un bottone per ogni modalità del mouse (ruotare la camera,
        # "volare" nella scena, muovere il modello, muovere il sole, muovere
        # l'ambiente IBL, selezionare un punto/giunto con un click). Il
        # pattern è identico per ognuno: crea il bottone, imposta il
        # padding (spaziatura interna) e collega la funzione da chiamare al click.
        self._arcball_button = gui.Button("Arcball")
        self._arcball_button.horizontal_padding_em = 0.5
        self._arcball_button.vertical_padding_em = 0
        self._arcball_button.set_on_clicked(self._set_mouse_mode_rotate)
        self._fly_button = gui.Button("Fly")
        self._fly_button.horizontal_padding_em = 0.5
        self._fly_button.vertical_padding_em = 0
        self._fly_button.set_on_clicked(self._set_mouse_mode_fly)
        self._model_button = gui.Button("Model")
        self._model_button.horizontal_padding_em = 0.5
        self._model_button.vertical_padding_em = 0
        self._model_button.set_on_clicked(self._set_mouse_mode_model)
        self._sun_button = gui.Button("Sun")
        self._sun_button.horizontal_padding_em = 0.5
        self._sun_button.vertical_padding_em = 0
        self._sun_button.set_on_clicked(self._set_mouse_mode_sun)
        self._ibl_button = gui.Button("Environment")
        self._ibl_button.horizontal_padding_em = 0.5
        self._ibl_button.vertical_padding_em = 0
        self._ibl_button.set_on_clicked(self._set_mouse_mode_ibl)
        self._pick_button = gui.Button("Pick")
        self._pick_button.horizontal_padding_em = 0.5
        self._pick_button.vertical_padding_em = 0
        self._pick_button.set_on_clicked(self._set_mouse_mode_pick)
        view_ctrls.add_child(gui.Label("Mouse controls"))
        # We want two rows of buttons, so make two horizontal layouts. We also
        # want the buttons centered, which we can do be putting a stretch item
        # as the first and last item. Stretch items take up as much space as
        # possible, and since there are two, they will each take half the extra
        # space, thus centering the buttons.
        # (In pratica: vogliamo due righe di bottoni centrate orizzontalmente;
        # aggiungendo uno "stretch" prima e dopo i bottoni, questi spazi
        # elastici si dividono lo spazio libero in parti uguali, centrando
        # visivamente il contenuto tra loro.)
        h = gui.Horiz(0.25 * em)  # row 1
        h.add_stretch()
        h.add_child(self._arcball_button)
        h.add_child(self._fly_button)
        h.add_child(self._model_button)
        h.add_stretch()
        view_ctrls.add_child(h)
        h = gui.Horiz(0.25 * em)  # row 2
        h.add_stretch()
        h.add_child(self._sun_button)
        h.add_child(self._ibl_button)
        h.add_child(self._pick_button)
        h.add_stretch()
        view_ctrls.add_child(h)

        # Checkbox per mostrare/nascondere lo "skymap" (sfondo ambientale)
        self._show_skybox = gui.Checkbox("Show skymap")
        self._show_skybox.set_on_checked(self._on_show_skybox)
        view_ctrls.add_fixed(separation_height)  # Aggiunge uno spazio verticale fisso, solo per estetica/leggibilità
        view_ctrls.add_child(self._show_skybox)

        # Selettore del colore di sfondo della scena
        self._bg_color = gui.ColorEdit()
        self._bg_color.set_on_value_changed(self._on_bg_color)

        # Una griglia a 2 colonne (etichetta + widget) per allineare "label" e controllo
        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("BG Color"))
        grid.add_child(self._bg_color)
        view_ctrls.add_child(grid)

        # Checkbox per mostrare/nascondere gli assi di riferimento X/Y/Z
        self._show_axes = gui.Checkbox("Show axes")
        self._show_axes.set_on_checked(self._on_show_axes)
        view_ctrls.add_fixed(separation_height)
        view_ctrls.add_child(self._show_axes)

        # Checkbox per mostrare/nascondere il pavimento a scacchiera
        self._show_ground = gui.Checkbox("Show ground")
        self._show_ground.set_on_checked(self._on_show_ground)
        view_ctrls.add_fixed(separation_height)
        view_ctrls.add_child(self._show_ground)

        # Menu a tendina con i profili di illuminazione predefiniti (definiti
        # in Settings.LIGHTING_PROFILES), più una voce "Custom" per quando
        # l'utente modifica manualmente i parametri di luce.
        self._profiles = gui.Combobox()
        for name in sorted(Settings.LIGHTING_PROFILES.keys()):
            self._profiles.add_item(name)
        self._profiles.add_item(Settings.CUSTOM_PROFILE_NAME)
        self._profiles.set_on_selection_changed(self._on_lighting_profile)
        view_ctrls.add_fixed(separation_height)
        view_ctrls.add_child(gui.Label("Lighting profiles"))
        view_ctrls.add_child(self._profiles)
        # NOTA: le sezioni vengono aggiunte al pannello tutte insieme in fondo
        # al costruttore, in ordine di importanza per l'utente (prima i
        # controlli dell'avatar, poi quelli di rendering) — cerca
        # "MONTAGGIO DEL PANNELLO" più avanti.

        # Sezione richiudibile "Advanced lighting": controlli più fini per
        # illuminazione ambientale (IBL) e luce solare direzionale.
        advanced = gui.CollapsableVert("Advanced lighting", 0,
                                       gui.Margins(em, 0, 0, 0))
        advanced.set_is_open(False)

        # Due checkbox per attivare/disattivare le due sorgenti di luce:
        # la mappa HDR ambientale (IBL) e il sole (luce direzionale).
        self._use_ibl = gui.Checkbox("HDR map")
        self._use_ibl.set_on_checked(self._on_use_ibl)
        self._use_sun = gui.Checkbox("Sun")
        self._use_sun.set_on_checked(self._on_use_sun)
        advanced.add_child(gui.Label("Light sources"))
        h = gui.Horiz(em)
        h.add_child(self._use_ibl)
        h.add_child(self._use_sun)
        advanced.add_child(h)

        # Menu a tendina per scegliere la mappa HDR (l'ambiente/skybox usato
        # come illuminazione): cerchiamo tutti i file "*_ibl.ktx" nella
        # cartella delle risorse di Open3D e li elenchiamo come opzioni.
        self._ibl_map = gui.Combobox()
        for ibl in glob.glob(gui.Application.instance.resource_path +
                             "/*_ibl.ktx"):

            self._ibl_map.add_item(os.path.basename(ibl[:-8]))
        self._ibl_map.selected_text = AppWindow.DEFAULT_IBL
        self._ibl_map.set_on_selection_changed(self._on_new_ibl)
        # Slider (a valori interi) per l'intensità della luce ambientale IBL
        self._ibl_intensity = gui.Slider(gui.Slider.INT)
        self._ibl_intensity.set_limits(0, 200000)
        self._ibl_intensity.set_on_value_changed(self._on_ibl_intensity)
        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("HDR map"))
        grid.add_child(self._ibl_map)
        grid.add_child(gui.Label("Intensity"))
        grid.add_child(self._ibl_intensity)
        advanced.add_fixed(separation_height)
        advanced.add_child(gui.Label("Environment"))
        advanced.add_child(grid)

        # Controlli per la luce solare: intensità (slider), direzione
        # (editor di vettore 3D) e colore (selettore colore).
        self._sun_intensity = gui.Slider(gui.Slider.INT)
        self._sun_intensity.set_limits(0, 200000)
        self._sun_intensity.set_on_value_changed(self._on_sun_intensity)
        self._sun_dir = gui.VectorEdit()
        self._sun_dir.set_on_value_changed(self._on_sun_dir)
        self._sun_color = gui.ColorEdit()
        self._sun_color.set_on_value_changed(self._on_sun_color)
        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("Intensity"))
        grid.add_child(self._sun_intensity)
        grid.add_child(gui.Label("Direction"))
        grid.add_child(self._sun_dir)
        grid.add_child(gui.Label("Color"))
        grid.add_child(self._sun_color)
        advanced.add_fixed(separation_height)
        advanced.add_child(gui.Label("Sun (Directional light)"))
        advanced.add_child(grid)


        # Sezione richiudibile "Material settings": scelta dello shader,
        # del materiale predefinito, del colore e della dimensione dei punti.
        material_settings = gui.CollapsableVert("Material settings", 0,
                                                gui.Margins(em, 0, 0, 0))
        material_settings.set_is_open(False)

        # Menu a tendina per scegliere lo shader/tipo di materiale (Lit,
        # Unlit, Normals, Depth): aggiungiamo le 4 voci definite in MATERIAL_NAMES.
        self._shader = gui.Combobox()
        self._shader.add_item(AppWindow.MATERIAL_NAMES[0])
        self._shader.add_item(AppWindow.MATERIAL_NAMES[1])
        self._shader.add_item(AppWindow.MATERIAL_NAMES[2])
        self._shader.add_item(AppWindow.MATERIAL_NAMES[3])
        self._shader.set_on_selection_changed(self._on_shader)
        # Menu a tendina per scegliere un materiale "prefab" (uno dei set
        # di proprietà fisiche definiti in Settings.PREFAB)
        self._material_prefab = gui.Combobox()
        for prefab_name in sorted(Settings.PREFAB.keys()):
            self._material_prefab.add_item(prefab_name)
        self._material_prefab.selected_text = Settings.DEFAULT_MATERIAL_NAME
        self._material_prefab.set_on_selection_changed(self._on_material_prefab)
        # Selettore colore per il materiale, e slider per la dimensione dei
        # punti (usato quando si visualizzano nuvole di punti/point cloud)
        self._material_color = gui.ColorEdit()
        self._material_color.set_on_value_changed(self._on_material_color)
        self._point_size = gui.Slider(gui.Slider.INT)
        self._point_size.set_limits(1, 10)
        self._point_size.set_on_value_changed(self._on_point_size)

        # Disponiamo tutti i controlli sopra in una griglia a 2 colonne (etichetta + widget)
        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("Type"))
        grid.add_child(self._shader)
        grid.add_child(gui.Label("Material"))
        grid.add_child(self._material_prefab)
        grid.add_child(gui.Label("Color"))
        grid.add_child(self._material_color)
        grid.add_child(gui.Label("Point size"))
        grid.add_child(self._point_size)
        material_settings.add_child(grid)


        # ----------------------------------- #
        # ------- BODY MODEL SETTINGS ------- #
        # ----------------------------------- #
        # Da qui in poi costruiamo la sezione più "applicativa" della GUI:
        # scelta del modello di corpo (SMPL/SMPLX/MANO/FLAME), del genere,
        # dei parametri di forma (betas), di espressione facciale e di posa.
        self.preload_body_models()  # Carica in anticipo (cache) i modelli di corpo da disco, per velocizzare i cambi successivi
        self._scene.scene.show_ground_plane(self.settings.show_ground, rendering.Scene.GroundPlane(0))  # Mostra subito il pavimento nella scena

        # ----------------------------------- #
        # -------- AVATAR PRESET PANEL ------ #
        # ----------------------------------- #
        # Sezione per scegliere un profilo di avatar predefinito (preset):
        # ogni preset è definito in presets.json da un insieme di misure
        # antropometriche realistiche (altezza, torace, vita, fianchi...) e
        # dalle betas corrispondenti, precalcolate con build_presets.py.
        # Dopo aver applicato un preset l'utente può comunque modificare
        # liberamente misure e betas: il profilo torna a "Custom".
        self.avatar_presets = load_presets()
        self._current_preset_id = 'custom'  # Id del preset attualmente applicato ('custom' = nessuno/modificato)
        self._applying_preset = False  # Flag per non segnare "Custom" mentre stiamo applicando noi stessi un preset

        self.preset_settings = gui.CollapsableVert("Avatar Preset", 0,
                                                   gui.Margins(em, 0, 0, 0))
        self.preset_settings.set_is_open(True)  # Parte aperta: è il punto di partenza suggerito all'utente

        self._preset_combo = gui.Combobox()
        self._preset_combo.add_item("Custom")  # Prima voce: nessun preset (avatar personalizzato)
        for preset in self.avatar_presets:
            self._preset_combo.add_item(preset['label'])
        self._preset_combo.set_on_selection_changed(self._on_preset_selected)

        # Etichetta che mostra la descrizione del preset selezionato
        self._preset_description = gui.Label("")

        self._apply_preset_btn = gui.Button("Apply Preset")
        self._apply_preset_btn.set_on_clicked(self._on_apply_preset)
        self._apply_preset_btn.enabled = False  # Attivo solo quando è selezionato un preset diverso da "Custom"

        # Bottone che riporta l'avatar allo stato iniziale (corpo "medio":
        # betas, posa, espressione e misure target tutti azzerati)
        self._reset_avatar_btn = gui.Button("Reset Avatar")
        self._reset_avatar_btn.set_on_clicked(self._on_reset_avatar)

        # Menu a tendina per la texture dell'avatar (aspetto estetico: pelle,
        # volto, capelli "dipinti" sull'immagine). Elenca le immagini trovate
        # in data/textures/; la texture viene applicata solo al modello SMPL,
        # l'unico per cui esiste il template UV ufficiale (smpl_uv.obj).
        self._smpl_triangle_uvs = None  # Cache del template UV (le UV non cambiano con betas/posa)
        self._texture_combo = gui.Combobox()
        self._texture_combo.add_item("None")
        if os.path.isdir(AppWindow.TEXTURE_DIR):
            for fname in sorted(os.listdir(AppWindow.TEXTURE_DIR)):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self._texture_combo.add_item(fname)
        self._texture_combo.set_on_selection_changed(self._on_texture_changed)

        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("Profile"))
        grid.add_child(self._preset_combo)
        grid.add_child(gui.Label("Texture"))
        grid.add_child(self._texture_combo)
        self.preset_settings.add_child(grid)
        self.preset_settings.add_child(self._preset_description)

        h = gui.Horiz(0.25 * em)
        h.add_stretch()
        h.add_child(self._apply_preset_btn)
        h.add_child(self._reset_avatar_btn)
        h.add_stretch()
        self.preset_settings.add_child(h)

        if not self.avatar_presets:
            # Il file presets.json manca o è vuoto: avvisiamo l'utente
            # (la GUI resta comunque utilizzabile senza preset)
            self.preset_settings.add_child(gui.Label("No presets found (run build_presets.py)"))

        self.model_settings = gui.CollapsableVert("Model settings", 0,
                                                  gui.Margins(em, 0, 0, 0))
        self.model_settings.set_is_open(True)  # Questa sezione parte aperta, essendo la più usata

        # Menu a tendina per scegliere il modello di corpo (SMPL, SMPLX, MANO, FLAME)
        self._body_model = gui.Combobox()
        for bm in AppWindow.BODY_MODEL_NAMES:
            self._body_model.add_item(bm)

        # Menu a tendina per scegliere il genere del modello (dipende dal
        # modello selezionato: inizialmente usiamo i generi del primo modello in lista)
        self._body_model_gender = gui.Combobox()
        for gender in AppWindow.BODY_MODEL_GENDERS[AppWindow.BODY_MODEL_NAMES[0]]:
            self._body_model_gender.add_item(gender)

        # ------- BODY MODEL BETAS SETTINGS ------- #
        # I "betas" sono i 10 parametri che controllano la forma/corporatura
        # del modello (altezza, peso, proporzioni, ecc.). Creiamo uno slider
        # e un'etichetta per ciascuno dei 10 parametri.
        self._body_beta_tensor = torch.zeros(1, 10)  # Tensore che tiene i valori correnti dei betas (inizialmente tutti a zero = corpo "medio")
        self._body_beta_sliders = []
        self._body_beta_labels = []
        for i in range(AppWindow.BODY_MODEL_N_BETAS[AppWindow.BODY_MODEL_NAMES[0]]):
            slider = gui.Slider(gui.Slider.DOUBLE)
            slider.set_limits(-5.0, 5.0)  # I betas tipicamente variano in un intervallo ragionevole attorno a 0 (deviazioni standard)
            self._body_beta_sliders.append(slider)
            label = gui.Label(AppWindow.BETA_LABELS[AppWindow.BODY_MODEL_NAMES[0]][i])
            self._body_beta_labels.append(label)
        self._body_beta_reset = gui.Button("Reset betas")  # Bottone per riportare tutti i betas a zero

        # Etichetta di testo che mostra i valori numerici correnti dei betas,
        # formattati con una cifra decimale e separati da virgola.
        self._body_beta_text = gui.Label("Betas")
        self._body_beta_text.text = f",".join(f'{x:.1f}'for x in self._body_beta_tensor[0].numpy().tolist())

        # ------- BODY MODEL EXPRESSION SETTINGS ------- #
        # I parametri di "espressione" (solo per modelli con volto, es. FLAME/SMPLX)
        # controllano le espressioni facciali. Menu a tendina per scegliere
        # quale delle 10 componenti modificare, più uno slider per il suo valore.
        self._body_model_exp_comp = gui.Combobox()
        for i in range(10):
            self._body_model_exp_comp.add_item(f'{i + 1:02d}')

        self._body_exp_val = gui.Slider(gui.Slider.DOUBLE)
        self._body_exp_val.set_limits(-5.0, 5.0)
        self._body_exp_tensor = torch.zeros(1, 10)  # Tensore con i valori correnti dei parametri di espressione (zero = neutro)
        self._body_exp_reset = gui.Button("Reset expression")

        self._body_exp_text = gui.Label("Expression")
        self._body_exp_text.text = f",".join(f'{x:.1f}' for x in self._body_exp_tensor[0].numpy().tolist())


        # ------- BODY MODEL POSE SETTINGS ------- #
        # Controlli per modificare la POSA (rotazione dei giunti) del corpo:
        # prima si sceglie quale gruppo di parametri modificare (es. body_pose,
        # left_hand_pose...), poi quale giunto specifico, infine si regolano
        # i tre angoli di rotazione X/Y/Z (in gradi) con tre slider.
        self._body_pose_comp = gui.Combobox()
        for k in AppWindow.POSE_PARAMS[AppWindow.BODY_MODEL_NAMES[0]].keys():
            self._body_pose_comp.add_item(k)

        self._body_pose_joint = gui.Combobox()  # Popolato dinamicamente in base al gruppo di posa scelto sopra

        # Slider per gli angoli di rotazione del giunto selezionato, in gradi,
        # con range -180/+180 (un giro completo attorno a ciascun asse)
        self._body_pose_joint_x = gui.Slider(gui.Slider.INT)
        self._body_pose_joint_x.set_limits(-180, 180)
        self._body_pose_joint_y = gui.Slider(gui.Slider.INT)
        self._body_pose_joint_y.set_limits(-180, 180)
        self._body_pose_joint_z = gui.Slider(gui.Slider.INT)
        self._body_pose_joint_z.set_limits(-180, 180)

        self._body_pose_reset = gui.Button("Reset pose")  # Riporta la posa alla T-pose (tutti gli angoli a zero)
        self._body_pose_ik = gui.Button("Run IK")  # Avvia il risolutore di Inverse Kinematics

        # Checkbox per mostrare/nascondere i marcatori dei giunti e le loro etichette testuali
        self._show_joints = gui.Checkbox("Show joints")
        self._show_joints.set_on_checked(self._on_show_joints)

        self._show_joint_labels = gui.Checkbox("Show joint labels")
        self._show_joint_labels.set_on_checked(self._on_show_joint_labels)

        # self._on_body_pose_comp(list(AppWindow.POSE_PARAMS[AppWindow.BODY_MODEL_NAMES[0]].keys())[0], 0)
        # Colleghiamo ora tutti i widget creati sopra alle rispettive funzioni
        # di callback, cioè le funzioni che verranno chiamate automaticamente
        # quando l'utente interagisce con quel controllo (cambia selezione,
        # sposta uno slider, clicca un bottone...).
        self._body_model.set_on_selection_changed(self._on_body_model)
        self._body_model_gender.set_on_selection_changed(self._on_body_model_gender)

        # Per ogni slider dei betas colleghiamo la stessa funzione di
        # callback, ma "catturando" l'indice i tramite un parametro di
        # default (idx=i) nella lambda: senza questo trucco, tutte le
        # lambda condividerebbero il valore FINALE di i nel ciclo invece
        # del valore che aveva al momento della creazione.
        for i, slider in enumerate(self._body_beta_sliders):
            slider.set_on_value_changed(lambda val, idx=i: self._on_body_beta_val(val, idx))
        self._body_beta_reset.set_on_clicked(self._on_body_beta_reset)

        self._body_exp_val.set_on_value_changed(self._on_body_exp_val)
        self._body_exp_reset.set_on_clicked(self._on_body_exp_reset)
        self._body_model_exp_comp.set_on_selection_changed(self._on_body_model_exp_comp)

        self._body_pose_comp.set_on_selection_changed(self._on_body_pose_comp)
        self._body_pose_joint.set_on_selection_changed(self._on_body_pose_joint)
        self._body_pose_joint_x.set_on_value_changed(self._on_body_pose_joint_x)
        self._body_pose_joint_y.set_on_value_changed(self._on_body_pose_joint_y)
        self._body_pose_joint_z.set_on_value_changed(self._on_body_pose_joint_z)
        self._body_pose_reset.set_on_clicked(self._on_body_pose_reset)

        self._body_pose_ik.set_on_clicked(self._on_run_ik)

        # Colleghiamo alla scena 3D le callback per gli eventi del mouse e
        # della tastiera, così da poter gestire click/trascinamenti e tasti
        # premuti mentre il cursore è sopra il widget 3D.
        self._scene.set_on_mouse(self._on_mouse_widget)
        self._scene.set_on_key(self._on_key_widget)

        # Da qui montiamo (aggiungiamo) tutti i widget creati sopra dentro i
        # rispettivi layout a griglia/orizzontali, nell'ordine in cui devono
        # apparire nel pannello "Model settings".
        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("Body Model"))
        grid.add_child(self._body_model)
        grid.add_child(gui.Label("Gender"))
        grid.add_child(self._body_model_gender)
        self.model_settings.add_child(grid)

        # Campo numerico per impostare un'altezza target (in cm): se
        # valorizzato, il programma calcola automaticamente il beta[0]
        # necessario a raggiungere quell'altezza (vedi load_body_model più avanti).
        height_grid = gui.VGrid(2, 0.25 * em)
        height_grid.add_child(gui.Label("Target Height (cm)"))
        self._target_height_val = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self._target_height_val.double_value = 0.0
        self._target_height_val.set_on_value_changed(self._on_target_height)
        height_grid.add_child(self._target_height_val)
        self.model_settings.add_child(height_grid)

        # Griglia con tutti gli slider dei betas, ciascuno affiancato dalla
        # sua etichetta descrittiva (es. "Stazza/Altezza")
        beta_grid = gui.VGrid(2, 0.25 * em)
        for i in range(len(self._body_beta_sliders)):
            beta_grid.add_child(self._body_beta_labels[i])
            beta_grid.add_child(self._body_beta_sliders[i])
        self.model_settings.add_child(beta_grid)

        # h = gui.Horiz(0.25 * em)  # row 1
        # h.add_child(self._body_beta_text)
        # self.model_settings.add_child(h)

        h = gui.Horiz(0.25 * em)  # row 2
        h.add_child(self._body_beta_reset)
        self.model_settings.add_child(h)

        # Controlli per l'espressione facciale: menu a tendina della
        # componente + slider del valore
        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("Exp Component"))
        grid.add_child(self._body_model_exp_comp)
        grid.add_child(gui.Label("Exp val:"))
        grid.add_child(self._body_exp_val)
        self.model_settings.add_child(grid)

        h = gui.Horiz(0.25 * em)  # row 2
        h.add_child(self._body_exp_reset)
        self.model_settings.add_child(h)

        # Riga con le due checkbox per mostrare giunti ed etichette dei giunti
        h = gui.Horiz(0.25 * em)  # row 2
        h.add_child(self._show_joints)
        h.add_child(self._show_joint_labels)
        self.model_settings.add_child(h)

        h = gui.Horiz(0.25 * em)  # row 3
        h.add_child(gui.Label("Pose Controls"))
        self.model_settings.add_child(h)

        # grid.add_child(gui.Label("Beta"))
        # grid.add_child(self._body_beta_text)
        # grid.add_child(gui.Label("reset"))
        # grid.add_child(self._body_beta_reset)
        # Griglia con i controlli di posa: componente, giunto, e i tre
        # slider degli angoli di rotazione X/Y/Z
        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("Pose comp:"))
        grid.add_child(self._body_pose_comp)
        grid.add_child(gui.Label("Joint id:"))
        grid.add_child(self._body_pose_joint)
        grid.add_child(gui.Label("rot_x"))
        grid.add_child(self._body_pose_joint_x)
        grid.add_child(gui.Label("rot_y"))
        grid.add_child(self._body_pose_joint_y)
        grid.add_child(gui.Label("rot_z"))
        grid.add_child(self._body_pose_joint_z)
        self.model_settings.add_child(grid)

        h = gui.Horiz(0.25 * em)  # row 2
        h.add_child(self._body_pose_reset)
        # h.add_child(gui.VectorEdit())
        self.model_settings.add_child(h)

        h = gui.Horiz(0.25 * em)  # row 2
        h.add_child(self._body_pose_ik)
        # h.add_child(gui.VectorEdit())
        self.model_settings.add_child(h)


        # ------- ANTHROPOMETRIC MEASUREMENTS PANEL ------- #
        # Questa è la sezione "Measurements", cuore della parte di
        # antropometria del progetto: permette di inserire misure corporee
        # target (in cm) e chiedere al programma di modificare i betas
        # dell'avatar in modo che le sue misure combacino con quelle inserite.
        self.measurement_settings = gui.CollapsableVert("Measurements", 0,
                                                         gui.Margins(em, 0, 0, 0))
        self.measurement_settings.set_is_open(True)  # Parte aperta perché è una funzionalità centrale del progetto

        self.measurement_settings.add_child(gui.Label("Target Measurements (cm)"))
        targets_grid = gui.VGrid(2, 0.25 * em)

        # Creiamo un campo numerico di input per ciascuna misura target:
        # le stesse misure mostrate in "Current Measurements", così ogni
        # valore visualizzato può anche essere usato come vincolo del
        # fitting (i campi lasciati a 0 non vincolano). Le teniamo in un
        # dizionario (self.target_inputs) indicizzato dalla chiave testuale
        # della misura, così da poterle leggere facilmente durante il fitting.
        self.target_inputs = {}
        target_keys = ["height", "chest circumference", "waist circumference", "hip circumference",
                       "inside leg height", "arm right length",
                       "neck circumference", "head circumference", "shoulder breadth"]
        target_labels = ["Height:", "Chest:", "Waist:", "Hip:",
                         "Inside Leg:", "Arm Length:",
                         "Neck:", "Head:", "Shoulder:"]

        for k, label_text in zip(target_keys, target_labels):
            targets_grid.add_child(gui.Label(label_text))
            num_edit = gui.NumberEdit(gui.NumberEdit.DOUBLE)
            num_edit.double_value = 0.0
            targets_grid.add_child(num_edit)
            self.target_inputs[k] = num_edit

        self.measurement_settings.add_child(targets_grid)

        # Bottone che avvia l'ottimizzazione: modifica i betas dell'avatar
        # per fargli assumere (il più possibile) le misure target inserite sopra.
        self._fit_measurements_btn = gui.Button("Fit Avatar to Targets")
        self._fit_measurements_btn.set_on_clicked(self._on_fit_measurements)

        h_fit = gui.Horiz(0.25 * em)
        h_fit.add_stretch()
        h_fit.add_child(self._fit_measurements_btn)
        h_fit.add_stretch()
        self.measurement_settings.add_child(h_fit)

        # Etichetta di stato del fitting: mostra "in corso..." durante
        # l'ottimizzazione e l'errore residuo massimo (in cm) alla fine,
        # così l'utente ha un riscontro concreto sulla qualità del fit.
        self._fit_status = gui.Label("")
        self.measurement_settings.add_child(self._fit_status)

        self.measurement_settings.add_fixed(separation_height)

        # Menu a tendina per scegliere quale misura visualizzare come
        # "overlay" (linea/circonferenza disegnata sopra la mesh) per
        # verificare visivamente dove viene presa quella misura.
        self.measurement_settings.add_child(gui.Label("Visible Overlay:"))
        self._visible_overlay = gui.Combobox()
        self._visible_overlay.add_item("None")
        for k in target_keys:
            self._visible_overlay.add_item(k)
        self._visible_overlay.set_on_selection_changed(self._on_visible_overlay_changed)
        self.measurement_settings.add_child(self._visible_overlay)

        self.measurement_settings.add_fixed(separation_height)

        # Griglia (popolata dinamicamente più avanti) che mostrerà le misure
        # ATTUALI calcolate sulla mesh corrente, per confronto con i target.
        self.measurement_settings.add_child(gui.Label("Current Measurements:"))
        self.current_measure_grid = gui.VGrid(2, 0.25 * em)
        self.current_measure_labels = {}
        # Creiamo un'etichetta "0.0 cm" per ciascuna misura (le stesse dei
        # campi target sopra) e le salviamo in un dizionario per poterle
        # aggiornare dinamicamente più avanti (in recalculate_and_update_measurements).
        for k, label_text in zip(target_keys, target_labels):
            self.current_measure_grid.add_child(gui.Label(label_text))
            lbl = gui.Label("0.0 cm")
            self.current_measure_grid.add_child(lbl)
            self.current_measure_labels[k] = lbl

        self.measurement_settings.add_child(self.current_measure_grid)

        # ------- MONTAGGIO DEL PANNELLO ------- #
        # Aggiungiamo qui tutte le sezioni al pannello laterale, in ordine di
        # importanza per l'utente: prima i controlli dell'avatar (preset,
        # misure, parametri del modello), poi quelli di rendering (materiali,
        # vista, illuminazione avanzata), che partono collassati.
        for section in (self.preset_settings, self.measurement_settings,
                        self.model_settings, material_settings, view_ctrls,
                        advanced):
            self._settings_panel.add_fixed(separation_height)
            self._settings_panel.add_child(section)

        # Info panel
        # Etichetta di testo generica usata per mostrare messaggi informativi
        # sovrapposti alla scena 3D (es. suggerimenti quando si seleziona un giunto).
        self.info = gui.Label("")
        self.info.visible = False

        # Etichetta 3D (ancorata a un punto nello spazio) usata per mostrare i
        # nomi dei giunti direttamente vicino alla loro posizione nella scena.
        self.joint_label_3d = gui.Label3D("", [0,0,0])
        self.joint_labels_3d_list = []
        # self.joint_label_3d.visible = False
        # ----

        # Normally our user interface can be children of all one layout (usually
        # a vertical layout), which is then the only child of the window. In our
        # case we want the scene to take up all the space and the settings panel
        # to go above it. We can do this custom layout by providing an on_layout
        # callback. The on_layout callback should set the frame
        # (position + size) of every child correctly. After the callback is
        # done the window will layout the grandchildren.
        # (In pratica: normalmente i widget si dispongono automaticamente in un
        # unico layout, ma qui vogliamo che la scena 3D occupi tutto lo spazio
        # e il pannello impostazioni "galleggi" sopra di essa: per farlo si usa
        # un callback di layout personalizzato, _on_layout, che imposta a mano
        # posizione e dimensione di ciascun widget figlio.)
        w.set_on_layout(self._on_layout)
        w.add_child(self._scene)
        w.add_child(self._settings_panel)
        w.add_child(self.info)
        # w.add_child(self.joint_label_3d)

        # ---- Menu ----
        # The menu is global (because the macOS menu is global), so only create
        # it once, no matter how many windows are created
        # (In pratica: su macOS il menu è unico per tutta l'applicazione, non
        # per singola finestra, quindi lo creiamo una sola volta.)
        if gui.Application.instance.menubar is None:
            if isMacOS:
                # Su macOS il primo menu è quello dell'applicazione stessa
                # (contiene tipicamente "About" e "Quit").
                app_menu = gui.Menu()
                app_menu.add_item("Help", AppWindow.MENU_ABOUT)
                app_menu.add_separator()
                app_menu.add_item("Quit", AppWindow.MENU_QUIT)
            # Menu "File": apertura file, esportazione immagine/OBJ, salvataggio parametri
            file_menu = gui.Menu()
            file_menu.add_item("Open", AppWindow.MENU_OPEN)
            file_menu.add_item("Export Current Image", AppWindow.MENU_EXPORT)
            file_menu.add_item("Export 3D Mesh (OBJ)", AppWindow.MENU_EXPORT_OBJ)
            file_menu.add_item("Save Model Params", AppWindow.MENU_SAVE)
            if not isMacOS:
                # Su Windows/Linux "Quit" va messo nel menu File (su macOS sta nel menu dell'app)
                file_menu.add_separator()
                file_menu.add_item("Quit", AppWindow.MENU_QUIT)
            # Menu "Settings": mostra/nasconde il pannello impostazioni
            settings_menu = gui.Menu()
            settings_menu.add_item("Model - Lighting - Materials",
                                   AppWindow.MENU_SHOW_SETTINGS)
            settings_menu.set_checked(AppWindow.MENU_SHOW_SETTINGS, True)
            # Menu "Help": informazioni sull'app
            help_menu = gui.Menu()
            help_menu.add_item("About", AppWindow.MENU_ABOUT)

            menu = gui.Menu()
            if isMacOS:
                # macOS will name the first menu item for the running application
                # (in our case, probably "Python"), regardless of what we call
                # it. This is the application menu, and it is where the
                # About..., Preferences..., and Quit menu items typically go.
                menu.add_menu("Example", app_menu)
                menu.add_menu("File", file_menu)
                menu.add_menu("Settings", settings_menu)
                # Don't include help menu unless it has something more than
                # About...
            else:
                menu.add_menu("File", file_menu)
                menu.add_menu("Settings", settings_menu)
                menu.add_menu("Help", help_menu)
            gui.Application.instance.menubar = menu

        # The menubar is global, but we need to connect the menu items to the
        # window, so that the window can call the appropriate function when the
        # menu item is activated.
        # (In pratica: anche se il menu è condiviso globalmente, ogni finestra
        # deve collegare le proprie funzioni di callback alle voci di menu.)
        w.set_on_menu_item_activated(AppWindow.MENU_OPEN, self._on_menu_open)
        w.set_on_menu_item_activated(AppWindow.MENU_EXPORT,
                                     self._on_menu_export)
        w.set_on_menu_item_activated(AppWindow.MENU_EXPORT_OBJ,
                                     self._on_export_obj_dialog)
        w.set_on_menu_item_activated(AppWindow.MENU_SAVE,
                                     self._on_save_dialog)
        w.set_on_menu_item_activated(AppWindow.MENU_QUIT, self._on_menu_quit)
        w.set_on_menu_item_activated(AppWindow.MENU_SHOW_SETTINGS,
                                     self._on_menu_toggle_settings_panel)
        w.set_on_menu_item_activated(AppWindow.MENU_ABOUT, self._on_menu_about)
        # ----

        # Applichiamo subito le impostazioni di default alla scena (colori,
        # luci, materiale) e carichiamo il primo modello di corpo della lista,
        # così l'utente vede subito un avatar all'avvio del programma.
        self._apply_settings()
        self._on_body_model(AppWindow.BODY_MODEL_NAMES[0], 0)

    def _apply_settings(self):
        # Questo metodo "traduce" i valori contenuti in self.settings
        # (l'oggetto Settings) in chiamate vere e proprie all'API di Open3D,
        # applicandoli alla scena 3D corrente. Viene richiamato ogni volta
        # che l'utente cambia un'impostazione tramite la GUI.
        bg_color = [
            self.settings.bg_color.red, self.settings.bg_color.green,
            self.settings.bg_color.blue, self.settings.bg_color.alpha
        ]
        self._scene.scene.set_background(bg_color)
        self._scene.scene.show_skybox(self.settings.show_skybox)
        self._scene.scene.show_axes(self.settings.show_axes)
        if self.settings.new_ibl_name is not None:
            self._scene.scene.scene.set_indirect_light(
                self.settings.new_ibl_name)
            # Clear new_ibl_name, so we don't keep reloading this image every
            # time the settings are applied.
            self.settings.new_ibl_name = None
        self._scene.scene.scene.enable_indirect_light(self.settings.use_ibl)
        self._scene.scene.scene.set_indirect_light_intensity(
            self.settings.ibl_intensity)
        sun_color = [
            self.settings.sun_color.red, self.settings.sun_color.green,
            self.settings.sun_color.blue
        ]
        self._scene.scene.scene.set_sun_light(self.settings.sun_dir, sun_color,
                                              self.settings.sun_intensity)
        self._scene.scene.scene.enable_sun_light(self.settings.use_sun)

        # Se è stato richiesto un cambio di materiale, lo applichiamo alla
        # scena e resettiamo il flag per non riapplicarlo inutilmente ad ogni frame.
        if self.settings.apply_material:
            self._scene.scene.update_material(self.settings.material)
            self.settings.apply_material = False

        # Sincronizziamo anche i widget della GUI con i valori attuali delle
        # impostazioni, così che se un'impostazione viene cambiata "da codice"
        # (es. caricando un profilo), i controlli grafici si aggiornino di conseguenza.
        self._bg_color.color_value = self.settings.bg_color
        self._show_skybox.checked = self.settings.show_skybox
        self._show_axes.checked = self.settings.show_axes
        self._show_ground.checked = self.settings.show_ground
        self._use_ibl.checked = self.settings.use_ibl
        self._use_sun.checked = self.settings.use_sun
        self._ibl_intensity.int_value = self.settings.ibl_intensity
        self._sun_intensity.int_value = self.settings.sun_intensity
        self._sun_dir.vector_value = self.settings.sun_dir
        self._sun_color.color_value = self.settings.sun_color
        self._material_prefab.enabled = (
            self.settings.material.shader == Settings.LIT)
        c = gui.Color(self.settings.material.base_color[0],
                      self.settings.material.base_color[1],
                      self.settings.material.base_color[2],
                      self.settings.material.base_color[3])
        self._material_color.color_value = c
        self._point_size.double_value = self.settings.material.point_size

    def _on_layout(self, layout_context):
        # The on_layout callback should set the frame (position + size) of every
        # child correctly. After the callback is done the window will layout
        # the grandchildren.
        # Questo metodo viene chiamato ogni volta che la finestra viene
        # ridimensionata: calcola manualmente la posizione e la dimensione
        # (il "frame", cioè un rettangolo x/y/larghezza/altezza) di ciascun
        # widget principale, per ottenere il layout desiderato (scena a
        # tutto schermo, pannello impostazioni ancorato a destra).
        r = self.window.content_rect  # Rettangolo che rappresenta l'intera area utile della finestra
        self._scene.frame = r  # La scena 3D occupa sempre tutta la finestra
        width = 22 * layout_context.theme.font_size  # Larghezza del pannello impostazioni, proporzionale al font
        height = min(
            r.height,
            self._settings_panel.calc_preferred_size(
                layout_context, gui.Widget.Constraints()).height)
        # Il pannello impostazioni viene posizionato appoggiato al bordo destro della finestra
        self._settings_panel.frame = gui.Rect(r.get_right() - width, r.y, width,
                                              height)

        # L'etichetta informativa "info" viene posizionata in basso a sinistra
        pref = self.info.calc_preferred_size(layout_context,
                                             gui.Widget.Constraints())
        self.info.frame = gui.Rect(r.x,
                                   r.get_bottom() - pref.height, pref.width,
                                   pref.height)

    # I sei metodi seguenti sono le callback dei bottoni "Mouse controls"
    # (Arcball/Fly/Model/Sun/Environment/Pick): ognuno imposta semplicemente
    # la modalità di controllo del mouse sulla scena 3D (SceneWidget.Controls),
    # cioè cosa succede quando trascini il mouse sulla vista.
    def _set_mouse_mode_rotate(self):
        # Modalità "Arcball": trascinare il mouse ruota la camera intorno alla scena
        self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)

    def _set_mouse_mode_fly(self):
        # Modalità "Fly": ci si muove nella scena come in un videogioco in prima persona
        self._scene.set_view_controls(gui.SceneWidget.Controls.FLY)

    def _set_mouse_mode_sun(self):
        # Modalità "Sun": trascinare il mouse cambia la direzione della luce solare
        self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_SUN)

    def _set_mouse_mode_ibl(self):
        # Modalità "Environment": trascinare il mouse ruota la mappa di illuminazione ambientale
        self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_IBL)

    def _set_mouse_mode_model(self):
        # Modalità "Model": trascinare il mouse ruota/muove il modello (la mesh) invece della camera
        self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_MODEL)

    def _set_mouse_mode_pick(self):
        # Modalità "Pick": un click seleziona un punto/oggetto nella scena (usato per scegliere un giunto)
        self._scene.set_view_controls(gui.SceneWidget.Controls.PICK_POINTS)

    def _on_bg_color(self, new_color):
        # Callback del selettore colore di sfondo: aggiorna le impostazioni e riapplica
        self.settings.bg_color = new_color
        self._apply_settings()

    def _on_show_skybox(self, show):
        # Callback della checkbox "Show skymap"
        self.settings.show_skybox = show
        self._apply_settings()

    def _on_show_axes(self, show):
        # Callback della checkbox "Show axes"
        self.settings.show_axes = show
        self._apply_settings()

    def _on_show_ground(self, show):
        # Callback della checkbox "Show ground"
        self.settings.show_ground = show
        self._apply_settings()

    def _on_show_joint_labels(self, show):
        # Callback della checkbox "Show joint labels": mostra o nasconde le
        # etichette testuali 3D con il nome di ciascun giunto, posizionate
        # accanto alla loro posizione nello spazio.
        if hasattr(self, "joint_label_3d"):
            self._scene.remove_3d_label(self.joint_label_3d)
        if hasattr(self, "joint_labels_3d_list"):
            # Rimuoviamo prima tutte le etichette eventualmente già disegnate,
            # per evitare di accumularle ad ogni chiamata
            for label3d in self.joint_labels_3d_list:
                self._scene.remove_3d_label(label3d)
        if show:
            joint_names = AppWindow.KEYPOINT_NAMES[self._body_model.selected_text]
            try:
                # Per ogni giunto aggiungiamo un'etichetta 3D con il suo nome,
                # posizionata alle coordinate del giunto stesso (AppWindow.JOINTS[i])
                for i in range(len(joint_names)):
                    self.joint_labels_3d_list.append(
                        self._scene.add_3d_label(AppWindow.JOINTS[i], joint_names[i])
                    )
            except Exception as e:
                print(e)
                import ipdb; ipdb.set_trace()
        else:
            if hasattr(self, "joint_labels_3d_list"):
                for label3d in self.joint_labels_3d_list:
                    self._scene.remove_3d_label(label3d)

    def _on_show_joints(self, show):
        # Callback della checkbox "Show joints": disegna (o rimuove) una
        # piccola sfera colorata in corrispondenza di ciascun giunto del
        # modello, per visualizzare lo scheletro del corpo.
        # Prima rimuoviamo eventuali sfere già presenti nella scena (fino a
        # un massimo "di sicurezza" di 150 giunti), identificate dal loro
        # nome univoco "__joints_i__".
        for i in range(150):
            if self._scene.scene.has_geometry(f"__joints_{i}__"):
                self._scene.scene.remove_geometry(f"__joints_{i}__")

        # Colori e raggi delle sfere: verde per il giunto selezionato, rosso
        # per gli altri; raggio diverso a seconda della parte del corpo
        # (mani e piedi più piccoli, testa ancora più piccola, corpo più grande).
        green = [0.3, 0.7, 0.3, 1.0]
        red = [0.7, 0.3, 0.3, 1.0]
        hand_radius = 0.01
        foot_radius = 0.01
        head_radius = 0.007
        body_radius = 0.05
        joint_names = AppWindow.KEYPOINT_NAMES[self._body_model.selected_text]

        # Materiale (colore + shader) per i giunti non selezionati (rosso)
        mat = rendering.MaterialRecord()
        mat.base_color = red
        mat.shader = "defaultLit"

        # Materiale per il giunto attualmente selezionato (verde)
        mat_selected = rendering.MaterialRecord()
        mat_selected.base_color = green
        mat_selected.shader = "defaultLit"

        joints = AppWindow.JOINTS
        if show:
            # logger.info('drawing joints')
            for i in range(joints.shape[0]):
                # Scegliamo il raggio della sfera in base a che tipo di
                # giunto è (mano, testa, piede o resto del corpo)
                radius = body_radius
                if joint_names[i] in LEFT_HAND_KEYPOINT_NAMES + RIGHT_HAND_KEYPOINT_NAMES:
                    radius = hand_radius
                elif joint_names[i] in HEAD_KEYPOINT_NAMES:
                    radius = head_radius
                elif joint_names[i] in FOOT_KEYPOINT_NAMES:
                    radius = foot_radius

                # Creiamo una piccola sfera 3D e la spostiamo (translate) nella
                # posizione del giunto i-esimo
                sg = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
                sg.compute_vertex_normals()  # Necessario per calcolare correttamente l'illuminazione della sfera
                # if i == AppWindow.SELECTED_JOINT:
                #     sg.paint_uniform_color(green)
                # else:
                #     sg.paint_uniform_color(red)
                sg.translate(joints[i])
                # Se questo è il giunto selezionato dall'utente, la coloriamo
                # di verde, altrimenti di rosso
                if (AppWindow.SELECTED_JOINT is not None) and (i == AppWindow.SELECTED_JOINT):
                    self._scene.scene.add_geometry(f"__joints_{i}__", sg, mat_selected)
                else:
                    self._scene.scene.add_geometry(f"__joints_{i}__", sg, mat)

            # logger.debug(AppWindow.JOINTS[20])
        else:
            # import ipdb; ipdb.set_trace()
            for i in range(150):
                if self._scene.scene.has_geometry(f"__joints_{i}__"):
                    self._scene.scene.remove_geometry(f"__joints_{i}__")

        # Aggiorniamo anche le etichette dei giunti (se erano attive) per
        # tenerle coerenti con lo stato appena disegnato/rimosso
        self._on_show_joint_labels(self._show_joint_labels.checked)
        # import ipdb; ipdb.set_trace()

    # ---- Callback per l'illuminazione avanzata ----
    # Ognuna di queste funzioni aggiorna un singolo parametro delle
    # impostazioni di luce, imposta il profilo su "Custom" (perché l'utente
    # ha modificato manualmente un valore, quindi non corrisponde più a
    # nessun profilo predefinito) e infine riapplica tutto alla scena.
    def _on_use_ibl(self, use):
        self.settings.use_ibl = use
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_use_sun(self, use):
        self.settings.use_sun = use
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_lighting_profile(self, name, index):
        # Se l'utente sceglie un profilo dal menu (diverso da "Custom"),
        # applichiamo tutti i parametri di quel profilo in un colpo solo.
        if name != Settings.CUSTOM_PROFILE_NAME:
            self.settings.apply_lighting_profile(name)
            self._apply_settings()

    def _on_new_ibl(self, name, index):
        # Cambia la mappa HDR (ambiente) usata per l'illuminazione IBL
        self.settings.new_ibl_name = gui.Application.instance.resource_path + "/" + name
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_ibl_intensity(self, intensity):
        self.settings.ibl_intensity = int(intensity)
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_sun_intensity(self, intensity):
        self.settings.sun_intensity = int(intensity)
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_sun_dir(self, sun_dir):
        # Chiamata anche quando l'utente trascina il sole direttamente
        # nella scena 3D (vedi set_on_sun_direction_changed nel costruttore)
        self.settings.sun_dir = sun_dir
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_sun_color(self, color):
        self.settings.sun_color = color
        self._apply_settings()

    def _on_shader(self, name, index):
        # Cambia lo shader/materiale attivo in base alla voce scelta nel
        # menu a tendina "Type" (Lit/Unlit/Normals/Depth)
        self.settings.set_material(AppWindow.MATERIAL_SHADERS[index])
        self._apply_settings()

    def _on_body_model(self, name, index):
        # Callback chiamata quando l'utente cambia il modello di corpo
        # (SMPL/SMPLX/MANO/FLAME) dal menu a tendina. Bisogna "resettare"
        # tutta la parte di GUI che dipende dal modello specifico: betas,
        # generi disponibili, componenti di posa, giunti, ecc.
        logger.info(f"Loading body model {name}-{index}")
        for slider in self._body_beta_sliders:
            slider.double_value = 0.0  # Azzeriamo tutti gli slider dei betas (corpo "medio" di default)
        if hasattr(self, '_target_height_val'):
            self._target_height_val.double_value = 0.0
        AppWindow.CAM_FIRST = True  # Segnaliamo che la camera va riposizionata automaticamente sul nuovo modello
        self.load_body_model(name)  # Carica effettivamente la mesh del nuovo modello
        # Aggiorniamo le etichette degli slider dei betas con quelle
        # specifiche del nuovo modello (es. FLAME ha "Dim. Testa" invece di "Stazza/Altezza")
        for i, label in enumerate(self._body_beta_labels):
            label.text = AppWindow.BETA_LABELS[name][i]
        self._body_model_gender.clear_items()

        # Ripopoliamo il menu a tendina dei generi con quelli validi per il nuovo modello
        for gender in AppWindow.BODY_MODEL_GENDERS[name]:
            self._body_model_gender.add_item(gender)

        # Ripopoliamo il menu a tendina delle componenti di posa (es.
        # body_pose, left_hand_pose...) specifiche del nuovo modello
        self._body_pose_comp.clear_items()
        for k in AppWindow.POSE_PARAMS[name].keys():
            self._body_pose_comp.add_item(k)

        # Ripopoliamo il menu a tendina dei singoli giunti per la prima
        # componente di posa selezionata, mostrando "indice-nome" (es. "0-root")
        self._body_pose_joint.clear_items()
        joint_names = AppWindow.JOINT_NAMES[name][self._body_pose_comp.selected_text]
        for i in range(AppWindow.POSE_PARAMS[name][self._body_pose_comp.selected_text].shape[1]):
            self._body_pose_joint.add_item(f'{i}-{joint_names[i]}')

        self._reset_rot_sliders()  # Riporta gli slider di rotazione X/Y/Z al valore del giunto ora selezionato
        AppWindow.SELECTED_JOINT = None  # Nessun giunto selezionato per il "Pick"/IK dopo il cambio modello
        self._on_show_joints(self._show_joints.checked)  # Ridisegna i marcatori dei giunti per il nuovo modello
        self._mark_custom_preset()  # Cambiare modello a mano invalida l'eventuale preset applicato

    def _on_body_model_gender(self, name, index):
        # Callback per il cambio di genere (neutral/male/female): ricarica
        # la mesh con lo stesso modello ma il genere scelto
        logger.info(f"Changing {self._body_model.selected_text} body model gender to {name}-{index}")
        for slider in self._body_beta_sliders:
            slider.double_value = 0.0
        self.load_body_model(self._body_model.selected_text, gender=name)
        self._reset_rot_sliders()
        self._on_show_joints(self._show_joints.checked)
        self._mark_custom_preset()  # Cambiare genere a mano invalida l'eventuale preset applicato
        # self._apply_settings()

    def _on_preset_selected(self, name, index):
        # Callback del menu a tendina dei preset: mostra la descrizione del
        # profilo scelto e abilita il bottone "Apply Preset" solo se è stato
        # selezionato un preset vero (l'indice 0 è "Custom", niente da applicare).
        if index <= 0:
            self._preset_description.text = ""
            self._apply_preset_btn.enabled = False
        else:
            preset = self.avatar_presets[index - 1]
            self._preset_description.text = preset.get('description', '')
            self._apply_preset_btn.enabled = True
        self.window.set_needs_layout()  # La descrizione può cambiare l'altezza della sezione

    def _mark_custom_preset(self):
        # Quando l'utente modifica manualmente la forma del corpo (slider
        # betas, fitting, cambio genere...) l'avatar non corrisponde più al
        # preset applicato: riportiamo il menu su "Custom" per riflettere
        # lo stato reale. Il flag _applying_preset evita di farlo mentre
        # siamo NOI a modificare i parametri durante l'applicazione di un preset.
        if not hasattr(self, '_preset_combo') or self._applying_preset:
            return
        self._current_preset_id = 'custom'
        self._preset_combo.selected_index = 0
        self._preset_description.text = ""
        self._apply_preset_btn.enabled = False

    def _on_apply_preset(self):
        # Callback del bottone "Apply Preset": applica il profilo selezionato.
        # In pratica: imposta modello e genere del preset, carica le betas
        # precalcolate (o le calcola al volo se build_presets.py non è stato
        # eseguito), riempie i campi target del pannello Measurements con le
        # misure del preset e ricarica la mesh. Da qui in poi l'utente può
        # modificare liberamente misure e parametri.
        index = self._preset_combo.selected_index
        if index <= 0:
            return
        preset = self.avatar_presets[index - 1]

        model_name = preset.get('model_type', 'smpl').upper()
        gender = preset.get('gender', 'neutral').lower()

        self._applying_preset = True
        try:
            # Se il preset usa un modello diverso da quello corrente, cambiamo
            # modello (questo ripopola generi, etichette betas e controlli di posa)
            if self._body_model.selected_text != model_name:
                self._body_model.selected_text = model_name
                self._on_body_model(model_name, AppWindow.BODY_MODEL_NAMES.index(model_name))
            self._body_model_gender.selected_text = gender

            betas = preset.get('betas')
            if betas is None:
                # Fallback: le betas non sono ancora state precalcolate.
                # Le calcoliamo al volo con la stessa funzione di fitting
                # condivisa (può richiedere qualche istante) e le teniamo in
                # cache in memoria per le prossime applicazioni dello stesso preset.
                self._update_label(f"Computing betas for '{preset['label']}'... Please wait.")
                model = AppWindow.PRELOADED_BODY_MODELS[f'{model_name.lower()}-{gender}']
                if not hasattr(self, 'measurers'):
                    self.measurers = {
                        'smpl': MeasureBody('smpl'),
                        'smplx': MeasureBody('smplx')
                    }
                measurer = self.measurers[model_name.lower()]
                measurer.gender = gender.upper()
                betas_np, _ = fit_betas_to_measurements(model, measurer, preset['target_measurements'])
                betas = [float(b) for b in betas_np]
                preset['betas'] = betas

            # Il vincolo di "Target Height" sovrascriverebbe il beta[0] del
            # preset al prossimo ricaricamento: lo azzeriamo
            self._target_height_val.double_value = 0.0

            # Applichiamo le betas del preset a tensore, slider ed etichetta
            self._body_beta_tensor[0] = torch.tensor(betas, dtype=torch.float32)
            for i in range(len(self._body_beta_sliders)):
                self._body_beta_sliders[i].double_value = float(betas[i])
            self._body_beta_text.text = f",".join(f'{x:.1f}' for x in self._body_beta_tensor[0].numpy().tolist())

            # Precompiliamo i campi target del pannello Measurements con le
            # misure del preset: così l'utente vede da quali misure è nato il
            # profilo e può modificarle e rifare il fit
            for k, num_edit in self.target_inputs.items():
                num_edit.double_value = float(preset['target_measurements'].get(k, 0.0))

            self._current_preset_id = preset.get('id', 'custom')
            self.load_body_model(model_name, gender=gender)
            self._update_label(f"Preset '{preset['label']}' applied")
        finally:
            self._applying_preset = False

    def _get_selected_texture_path(self):
        # Ritorna il percorso dell'immagine texture selezionata nel menu a
        # tendina "Texture", o None se non c'è selezione (o il menu non
        # esiste ancora durante la costruzione della GUI).
        if not hasattr(self, '_texture_combo'):
            return None
        name = self._texture_combo.selected_text
        if not name or name == "None":
            return None
        path = os.path.join(AppWindow.TEXTURE_DIR, name)
        return path if os.path.exists(path) else None

    def _get_smpl_triangle_uvs(self):
        # Carica (una sola volta) il template UV ufficiale di SMPL e lo tiene
        # in cache: le coordinate UV dipendono solo dalla topologia della
        # mesh, non da betas o posa, quindi non vanno mai ricalcolate.
        # Usa False come valore sentinella per "già provato, file mancante",
        # così non ritentiamo il caricamento a ogni frame.
        if self._smpl_triangle_uvs is None:
            uv_path = os.path.join(AppWindow.TEXTURE_DIR, AppWindow.UV_TEMPLATE_FILE)
            if os.path.exists(uv_path):
                self._smpl_triangle_uvs = load_obj_triangle_uvs(uv_path)
                if self._smpl_triangle_uvs is None:
                    self._smpl_triangle_uvs = False
            else:
                logger.warning(f'SMPL UV template not found at {uv_path} '
                               f'(download smpl_uv.obj from smpl.is.tue.mpg.de)')
                self._smpl_triangle_uvs = False
        return None if self._smpl_triangle_uvs is False else self._smpl_triangle_uvs

    def _on_texture_changed(self, name, index):
        # Callback del menu "Texture": basta ricaricare la mesh, che
        # applicherà (o toglierà) la texture selezionata
        if self._body_model.selected_text != 'SMPL' and name != "None":
            self._update_label("Textures are only available for the SMPL model")
        self.load_body_model(
            self._body_model.selected_text,
            gender=self._body_model_gender.selected_text,
        )

    def _on_reset_avatar(self):
        # Bottone "Reset Avatar": riporta l'avatar completo allo stato
        # iniziale. Azzera la posa (tutti i gruppi, non solo quello
        # selezionato come fa "Reset pose"), l'espressione facciale e infine
        # delega a _on_body_beta_reset l'azzeramento di betas, altezza target
        # e misure target (che ricarica anche la mesh).
        bm = self._body_model.selected_text
        for bp in AppWindow.POSE_PARAMS[bm]:
            AppWindow.POSE_PARAMS[bm][bp] = torch.zeros_like(AppWindow.POSE_PARAMS[bm][bp])
        self._reset_rot_sliders()
        self._body_exp_tensor = torch.zeros(1, 10)
        self._body_exp_text.text = f",".join(f'{x:.1f}' for x in self._body_exp_tensor[0].numpy().tolist())
        self._body_exp_val.double_value = 0.0
        self._mark_custom_preset()
        if hasattr(self, '_fit_status'):
            self._fit_status.text = ""
        self._on_body_beta_reset()
        self._update_label("Avatar reset to default body")

    def _on_target_height(self, val):
        # Callback del campo "Target Height (cm)": ricarica il modello, che
        # internamente (in load_body_model) userà il valore per calcolare
        # automaticamente il beta[0] necessario a ottenere quell'altezza.
        self.load_body_model(
            self._body_model.selected_text,
            gender=self._body_model_gender.selected_text,
        )

    def _on_body_beta_val(self, val, idx):
        # Callback chiamata quando l'utente sposta uno degli slider dei
        # betas: aggiorna il tensore dei betas nella posizione idx e ricarica
        # la mesh con i nuovi parametri di forma.
        self._body_beta_tensor[0, idx] = float(val)
        if idx == 0 and hasattr(self, '_target_height_val') and self._target_height_val.double_value > 0.0:
            # Se l'utente muove manualmente lo slider del beta[0] (che
            # controlla anche l'altezza), disattiviamo il vincolo di
            # "altezza target" per evitare che venga sovrascritto subito dopo.
            self._target_height_val.double_value = 0.0
        self._mark_custom_preset()  # La forma è stata modificata a mano: non corrisponde più al preset
        self._body_beta_text.text = f",".join(f'{x:.1f}' for x in self._body_beta_tensor[0].numpy().tolist())
        self.load_body_model(
            self._body_model.selected_text,
            gender=self._body_model_gender.selected_text,
        )
        # self._on_show_joints(self._show_joints.checked)

    def _on_body_exp_val(self, val):
        # Callback per lo slider del valore di espressione facciale: aggiorna
        # la componente scelta nel tensore delle espressioni e ricarica la mesh
        self._body_exp_tensor[0, int(self._body_model_exp_comp.selected_text)-1] = float(val)
        self._body_exp_text.text = f",".join(f'{x:.1f}' for x in self._body_exp_tensor[0].numpy().tolist())
        self.load_body_model(
            self._body_model.selected_text,
            gender=self._body_model_gender.selected_text,
        )
        # self._on_show_joints(self._show_joints.checked)

    def _on_body_pose_joint(self, name, index):
        # Callback quando si sceglie un giunto diverso dal menu a tendina:
        # aggiorna gli slider di rotazione mostrando i valori correnti di quel giunto
        self._reset_rot_sliders()

    def _on_body_pose_joint_x(self, val):
        # Callback dello slider "rot_x": aggiorna la rotazione X del giunto
        # selezionato mantenendo Y e Z correnti, poi converte l'angolo di
        # Eulero (in gradi) in axis-angle (la rappresentazione richiesta dai
        # modelli SMPL/SMPLX/MANO/FLAME) e aggiorna il tensore di posa.
        bm = self._body_model.selected_text
        bp = self._body_pose_comp.selected_text
        ji = int(self._body_pose_joint.selected_text.split('-')[0])
        euler_angle = [val, self._body_pose_joint_y.int_value, self._body_pose_joint_z.int_value]
        # Conversione da angoli di Eulero (rotazioni successive attorno a x,
        # y, z) alla rappresentazione "axis-angle" (rotvec): un unico vettore
        # 3D la cui direzione è l'asse di rotazione e la cui lunghezza è
        # l'angolo di rotazione. È il formato numerico atteso dai modelli SMPL.
        axis_angle = R.Rotation.from_euler('xyz', euler_angle, degrees=True).as_rotvec()
        AppWindow.POSE_PARAMS[bm][bp][0, ji] = torch.from_numpy(axis_angle)

        self.load_body_model(
            self._body_model.selected_text,
            gender=self._body_model_gender.selected_text,
        )
        # self._on_show_joints(self._show_joints.checked)

    def _on_body_pose_joint_y(self, val):
        # Identico a _on_body_pose_joint_x ma per l'asse Y
        bm = self._body_model.selected_text
        bp = self._body_pose_comp.selected_text
        ji = int(self._body_pose_joint.selected_text.split('-')[0])
        euler_angle = [self._body_pose_joint_x.int_value, val, self._body_pose_joint_z.int_value]
        axis_angle = R.Rotation.from_euler('xyz', euler_angle, degrees=True).as_rotvec()
        AppWindow.POSE_PARAMS[bm][bp][0, ji] = torch.from_numpy(axis_angle)

        self.load_body_model(
            self._body_model.selected_text,
            gender=self._body_model_gender.selected_text,
        )
        # self._on_show_joints(self._show_joints.checked)

    def _on_body_pose_joint_z(self, val):
        # Identico a _on_body_pose_joint_x ma per l'asse Z
        bm = self._body_model.selected_text
        bp = self._body_pose_comp.selected_text
        ji = int(self._body_pose_joint.selected_text.split('-')[0])
        euler_angle = [self._body_pose_joint_x.int_value, self._body_pose_joint_y.int_value, val]
        axis_angle = R.Rotation.from_euler('xyz', euler_angle, degrees=True).as_rotvec()
        AppWindow.POSE_PARAMS[bm][bp][0, ji] = torch.from_numpy(axis_angle)

        self.load_body_model(
            self._body_model.selected_text,
            gender=self._body_model_gender.selected_text,
        )
        # self._on_show_joints(self._show_joints.checked)

    def _on_body_model_exp_comp(self, name, index):
        # Quando si sceglie una diversa componente di espressione dal menu,
        # mostriamo nello slider il suo valore corrente (senza modificarlo)
        self._body_exp_val.double_value = self._body_exp_tensor[0, index].item()

    def _on_body_pose_comp(self, name, index):
        # Quando si sceglie un diverso gruppo di parametri di posa (es. da
        # "body_pose" a "left_hand_pose"), ripopoliamo il menu dei giunti
        # disponibili per quel gruppo specifico
        self._body_pose_joint.clear_items()
        joint_names = AppWindow.JOINT_NAMES[self._body_model.selected_text][name]
        for i in range(AppWindow.POSE_PARAMS[self._body_model.selected_text][name].shape[1]):
            self._body_pose_joint.add_item(f'{i}-{joint_names[i]}')
        self._reset_rot_sliders()

    def _on_body_beta_reset(self):
        # Bottone "Reset betas": azzera il tensore dei betas, gli slider
        # corrispondenti, l'eventuale altezza target e le misure target
        # inserite, poi ricarica il modello (tornando al corpo "medio")
        self._body_beta_tensor = torch.zeros(1, 10)
        self._body_beta_text.text = f",".join(f'{x:.1f}' for x in self._body_beta_tensor[0].numpy().tolist())
        for slider in self._body_beta_sliders:
            slider.double_value = 0.0
        if hasattr(self, '_target_height_val'):
            self._target_height_val.double_value = 0.0
        if hasattr(self, 'target_inputs'):
            for k in self.target_inputs.keys():
                self.target_inputs[k].double_value = 0.0
        self._mark_custom_preset()  # Il corpo "medio" non corrisponde a nessun preset
        self.load_body_model(
            self._body_model.selected_text,
            gender=self._body_model_gender.selected_text,
        )

    def _on_body_exp_reset(self):
        # Bottone "Reset expression": azzera i parametri di espressione facciale
        self._body_exp_tensor = torch.zeros(1, 10)
        self._body_exp_text.text = f",".join(f'{x:.1f}' for x in self._body_exp_tensor[0].numpy().tolist())
        self._body_exp_val.double_value = 0.0
        self.load_body_model(
            self._body_model.selected_text,
            gender=self._body_model_gender.selected_text,
        )

    def _on_body_pose_reset(self):
        # Bottone "Reset pose": riporta a zero SOLO il gruppo di parametri di
        # posa attualmente selezionato (es. solo "body_pose", non le mani)
        bm = self._body_model.selected_text
        bp = self._body_pose_comp.selected_text
        AppWindow.POSE_PARAMS[bm][bp] = torch.zeros_like(AppWindow.POSE_PARAMS[bm][bp])
        self._reset_rot_sliders()
        self.load_body_model(
            self._body_model.selected_text,
            gender=self._body_model_gender.selected_text,
        )

    def _on_key_widget(self, event):
        # Callback per gli eventi da tastiera mentre il focus è sulla scena
        # 3D: se un giunto è selezionato (tramite "Pick"), i tasti numerici
        # da 1 a 6 lo spostano di un piccolo passo lungo i tre assi
        # (1/2 = -X/+X, 3/4 = -Y/+Y, 5/6 = -Z/+Z). Utile per "correggere" a
        # mano la posizione di un giunto prima di lanciare l'IK.
        key = gui.KeyName(event.key.real).name
        step = 0.01
        # logger.debug(f"key {key} is pressed")
        if (self._show_joints.checked) and \
                (AppWindow.SELECTED_JOINT is not None) and \
                (key in ('ONE', 'TWO', 'THREE', 'FOUR', 'FIVE', 'SIX')):
            if key == 'ONE':
                transl = np.array([-step, 0.0, 0.0])
            elif key == 'TWO':
                transl = np.array([step, 0.0, 0.0])
            elif key == 'THREE':
                transl = np.array([0.0, -step, 0.0])
            elif key == 'FOUR':
                transl = np.array([0.0, step, 0.0])
            elif key == 'FIVE':
                transl = np.array([0.0, 0.0, -step])
            elif key == 'SIX':
                transl = np.array([0.0, 0.0, step])

            AppWindow.JOINTS[AppWindow.SELECTED_JOINT] = AppWindow.JOINTS[AppWindow.SELECTED_JOINT] + transl
            self._on_show_joints(show=True)  # Ridisegna i giunti nella nuova posizione
            return gui.Widget.EventCallbackResult.HANDLED  # Segnala a Open3D che l'evento è stato gestito e non deve propagarsi oltre
        return gui.Widget.EventCallbackResult.IGNORED  # Altrimenti lasciamo che Open3D gestisca il tasto normalmente

    def _on_mouse_widget(self, event):
        # We could override BUTTON_DOWN without a modifier, but that would
        # interfere with manipulating the scene.
        # (Qui gestiamo i click del mouse sulla scena 3D: in particolare, il
        # click con un tasto modificatore serve a selezionare un giunto senza
        # interferire con la normale rotazione/pan della camera.)

        # self.joint_label_3d.text = ""
        # if self._show_joints.checked:
        #     mouse_pos = self._scene.scene.camera.unproject(
        #         event.x, (self._scene.frame.height - event.y), 0.1, self._scene.frame.width,
        #         self._scene.frame.height)
        #     # logger.debug(mouse_pos)
        #     label_idx = np.argmin(((AppWindow.JOINTS - mouse_pos) ** 2).sum(1))
        #     label_text = AppWindow.KEYPOINT_NAMES[self._body_model.selected_text][label_idx]
        #     label_pos = AppWindow.JOINTS[label_idx]
        #     self.joint_label_3d.text = label_text
        #     self.joint_label_3d.position = label_pos
        #     logger.debug(label_text, label_pos)
            # self._scene.add_3d_label(label_pos, label_text)

        # Gestiamo il click SOLO se: è un click "a bottone premuto", è
        # tenuto premuto anche il tasto CTRL (per non confliggere con la
        # normale rotazione della camera) e sono visibili i giunti.
        if event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.is_modifier_down(
                gui.KeyModifier.CTRL) and self._show_joints.checked:
            # x = event.x - self._scene.frame.x
            # y = event.y - self._scene.frame.y
            # logger.debug(f'Clicked point x: {x}, y: {y}')

            # Funzione chiamata in modo asincrono da Open3D non appena è
            # disponibile l'immagine di profondità (depth) della scena
            # renderizzata: ci serve per capire A CHE PUNTO 3D corrisponde
            # il pixel su cui l'utente ha cliccato in 2D.
            def depth_callback(depth_image):
                # Coordinates are expressed in absolute coordinates of the
                # window, but to dereference the image correctly we need them
                # relative to the origin of the widget. Note that even if the
                # scene widget is the only thing in the window, if a menubar
                # exists it also takes up space in the window (except on macOS).
                x = event.x # - self._scene.frame.x
                y = event.y # - self._scene.frame.y
                # logger.debug(f'Clicked point x: {x}, y: {y}')
                # Note that np.asarray() reverses the axes.
                # Leggiamo il valore di profondità nel pixel cliccato
                depth = np.asarray(depth_image)[y, x]
                # import skimage.io as io
                # io.imsave('depth_img.jpg', np.asarray(depth_image))

                if depth == 1.0:  # clicked on nothing (i.e. the far plane)
                    # Se la profondità è 1.0 significa che il click è "andato
                    # a vuoto" (non ha colpito nulla, solo lo sfondo lontano)
                    text = ""
                else:
                    # world = self._scene.scene.camera.unproject(
                    #     event.x, event.y, depth, self._scene.frame.width,
                    #     self._scene.frame.height)
                    # "unproject" converte le coordinate 2D del click + la
                    # profondità letta in una posizione 3D reale nella scena
                    # (l'operazione inversa della proiezione prospettica
                    # usata per disegnare la scena 3D su uno schermo 2D).
                    world = self._scene.scene.camera.unproject(
                        x, (self._scene.frame.height - y), depth, self._scene.frame.width,
                        self._scene.frame.height)

                    # logger.debug(f'cam pose {self._scene.scene.camera.get_model_matrix()[:3, 3].tolist()}')
                    # logger.debug(world)
                    # cam_x = self._scene.scene.camera.get_model_matrix()[:3, 3][0]
                    # cam_y = self._scene.scene.camera.get_model_matrix()[:3, 3][1]
                    # text = "({:.3f}, {:.3f}, {:.3f})".format(
                    #     world[0], world[1], world[2])
                    # logger.debug(f'Clicked {text}')

                    # find the closest joint to the clicked pos
                    # Troviamo il giunto più vicino al punto 3D cliccato,
                    # calcolando la distanza euclidea al quadrato da ogni
                    # giunto e prendendo l'indice di quello minimo (argmin).
                    dist = ((AppWindow.JOINTS - np.array([world[0], world[1], world[2]]))**2).sum(1)
                    AppWindow.SELECTED_JOINT = np.argmin(dist)
                    # logger.debug(AppWindow.SELECTED_JOINT)
                    # import ipdb; ipdb.set_trace()
                    jn = AppWindow.KEYPOINT_NAMES[self._body_model.selected_text][AppWindow.SELECTED_JOINT]
                    self._update_label(f'{self._body_model.selected_text} joint "{jn}" selected')
                    # Aggiorniamo l'etichetta 3D che mostra il nome del giunto appena selezionato
                    self._scene.remove_3d_label(self.joint_label_3d)
                    self.joint_label_3d = self._scene.add_3d_label(
                        AppWindow.JOINTS[AppWindow.SELECTED_JOINT],
                        AppWindow.KEYPOINT_NAMES[self._body_model.selected_text][AppWindow.SELECTED_JOINT]
                    )
                    # self.joint_label_3d.text = jn
                    # self.joint_label_3d.position = AppWindow.JOINTS[AppWindow.SELECTED_JOINT]
                    self._on_show_joints(show=True)  # Ridisegna i giunti per evidenziare in verde quello appena selezionato

                # This is not called on the main thread, so we need to
                # post to the main thread to safely access UI items.
                # def update_label():
                #     self.info.text = text
                #     self.info.visible = (text != "")
                #     # We are sizing the info label to be exactly the right size,
                #     # so since the text likely changed width, we need to
                #     # re-layout to set the new frame.
                #     self.window.set_needs_layout()

                # gui.Application.instance.post_to_main_thread(
                #     self.window, update_label)

            # Chiediamo a Open3D di renderizzare l'immagine di profondità e
            # di chiamare "depth_callback" non appena è pronta (operazione asincrona)
            self._scene.scene.scene.render_to_depth_image(depth_callback)
            return gui.Widget.EventCallbackResult.HANDLED
        return gui.Widget.EventCallbackResult.IGNORED

    def _update_label(self, text):
        # Funzione di utilità per mostrare un messaggio testuale nell'etichetta
        # informativa in basso a sinistra, forzando poi un ricalcolo del layout
        # (perché la larghezza del testo può essere cambiata)
        self.info.text = text
        self.info.visible = (text != "")
        # We are sizing the info label to be exactly the right size,
        # so since the text likely changed width, we need to
        # re-layout to set the new frame.
        self.window.set_needs_layout()

    def _on_run_ik(self):
        # Callback del bottone "Run IK": avvia la risoluzione di Inverse
        # Kinematics, cioè calcola quali angoli di posa fanno sì che i
        # giunti del corpo (in particolare quelli spostati manualmente con
        # i tasti numerici) raggiungano le posizioni desiderate.
        bm = self._body_model.selected_text
        bp = self._body_pose_comp.selected_text

        # L'IK qui implementato funziona solo per SMPL/SMPLX sul gruppo "body_pose"
        if not ((bm in ['SMPL', 'SMPLX']) and (bp in ('body_pose'))):
            logger.warning('IK is not implemented for this body model')
            return 0

        gender = self._body_model_gender.selected_text
        init_pose = copy.deepcopy(AppWindow.POSE_PARAMS[bm][bp])  # Copia della posa attuale, usata come punto di partenza dell'ottimizzazione

        # Prendiamo le posizioni correnti dei primi 22 giunti come "target"
        # da raggiungere (i giunti eventualmente spostati con i tasti numerici)
        target_keypoints = AppWindow.JOINTS[:22][None]
        target_keypoints = torch.from_numpy(target_keypoints).float()
        # Chiamiamo il nostro risolutore IK (in simple_ik.py): ottimizza
        # iterativamente gli angoli di posa per minimizzare la distanza tra
        # i giunti del modello e le posizioni target, fino a un massimo di 50 iterazioni.
        opt_params = simple_ik_solver(
            model=AppWindow.PRELOADED_BODY_MODELS[f'{bm.lower()}-{gender.lower()}'],
            target=target_keypoints, init=init_pose, device='cpu',
            max_iter=50, transl=AppWindow.BODY_TRANSL,
            betas=self._body_beta_tensor,
        )
        opt_params = opt_params.requires_grad_(False)  # Non ci serve più calcolare gradienti su questo tensore
        # import ipdb; ipdb.set_trace()
        AppWindow.POSE_PARAMS[bm][bp] = opt_params.reshape(1, -1, 3)  # Salviamo la nuova posa ottimizzata

        self.load_body_model(
            self._body_model.selected_text,
            gender=self._body_model_gender.selected_text,
        )

    def _reset_rot_sliders(self):
        # Riporta a zero gli slider di rotazione (usato quando si cambia
        # giunto o gruppo di posa, per non mostrare valori del giunto precedente)
        self._body_pose_joint_x.int_value = 0
        self._body_pose_joint_y.int_value = 0
        self._body_pose_joint_z.int_value = 0

    def _on_material_prefab(self, name, index):
        # Callback del menu a tendina "Material": applica il prefab scelto
        self.settings.apply_material_prefab(name)
        self.settings.apply_material = True
        self._apply_settings()

    def _on_material_color(self, color):
        # Callback del selettore colore del materiale
        self.settings.material.base_color = [
            color.red, color.green, color.blue, color.alpha
        ]
        self.settings.apply_material = True
        self._apply_settings()

    def _on_point_size(self, size):
        # Callback dello slider "Point size" (dimensione dei punti, rilevante per point cloud)
        self.settings.material.point_size = int(size)
        self.settings.apply_material = True
        self._apply_settings()

    def _on_menu_open(self):
        # Callback della voce di menu "File > Open": mostra una finestra di
        # dialogo per scegliere un file di mesh/point cloud da caricare
        dlg = gui.FileDialog(gui.FileDialog.OPEN, "Choose file to load",
                             self.window.theme)
        # Aggiungiamo diversi filtri di estensione, per aiutare l'utente a
        # trovare più facilmente i file del tipo giusto (mesh triangolari o
        # nuvole di punti), oltre a un filtro per ciascuna estensione singola.
        dlg.add_filter(
            ".ply .stl .fbx .obj .off .gltf .glb",
            "Triangle mesh files (.ply, .stl, .fbx, .obj, .off, "
            ".gltf, .glb)")
        dlg.add_filter(
            ".xyz .xyzn .xyzrgb .ply .pcd .pts",
            "Point cloud files (.xyz, .xyzn, .xyzrgb, .ply, "
            ".pcd, .pts)")
        dlg.add_filter(".ply", "Polygon files (.ply)")
        dlg.add_filter(".stl", "Stereolithography files (.stl)")
        dlg.add_filter(".fbx", "Autodesk Filmbox files (.fbx)")
        dlg.add_filter(".obj", "Wavefront OBJ files (.obj)")
        dlg.add_filter(".off", "Object file format (.off)")
        dlg.add_filter(".gltf", "OpenGL transfer files (.gltf)")
        dlg.add_filter(".glb", "OpenGL binary transfer files (.glb)")
        dlg.add_filter(".xyz", "ASCII point cloud files (.xyz)")
        dlg.add_filter(".xyzn", "ASCII point cloud with normals (.xyzn)")
        dlg.add_filter(".xyzrgb",
                       "ASCII point cloud files with colors (.xyzrgb)")
        dlg.add_filter(".pcd", "Point Cloud Data files (.pcd)")
        dlg.add_filter(".pts", "3D Points files (.pts)")
        dlg.add_filter("", "All files")

        # A file dialog MUST define on_cancel and on_done functions
        # (Ogni finestra di dialogo file deve avere una funzione da chiamare
        # se l'utente annulla, e una da chiamare se conferma la scelta.)
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_load_dialog_done)
        self.window.show_dialog(dlg)

    def _on_save_dialog(self):
        # Callback di "File > Save Model Params": mostra il dialogo per
        # scegliere dove salvare i parametri correnti del modello (betas, posa, ecc.)
        dlg = gui.FileDialog(gui.FileDialog.SAVE, "Choose file to save",
                             self.window.theme)
        dlg.set_on_cancel(self._on_save_dialog_cancel)
        dlg.set_on_done(self._on_save_dialog_done)
        self.window.show_dialog(dlg)

    def _on_save_dialog_cancel(self):
        self.window.close_dialog()

    def _on_save_dialog_done(self, filename):
        # Costruisce un dizionario con tutti i parametri correnti del
        # modello (betas, espressione, genere, tipo di modello, giunti e i
        # parametri di posa specifici del modello) e lo salva su disco con
        # joblib (una libreria pensata per serializzare in modo efficiente
        # oggetti Python, inclusi tensori numerici).
        self.window.close_dialog()
        output_dict = {
            'betas': self._body_beta_tensor,
            'expression': self._body_exp_tensor,
            'gender': self._body_model_gender.selected_text,
            'body_model': self._body_model.selected_text,
            'joints': AppWindow.JOINTS,
        }
        output_dict.update(AppWindow.POSE_PARAMS[self._body_model.selected_text])
        logger.debug(f'Saving output to {filename}')
        joblib.dump(output_dict, filename)

        # Accanto al file joblib salviamo anche una versione JSON "portabile"
        # dei parametri principali (stesso schema di presets.json): è il
        # formato che l'applicazione Unity legge per ricostruire lo stesso
        # avatar tramite i blendshapes del pacchetto SMPL.
        model_key = self._body_model.selected_text.lower()
        measurements = {}
        if hasattr(self, 'measurers') and model_key in self.measurers:
            # Le misure correnti sono già state calcolate all'ultimo
            # aggiornamento della mesh (recalculate_and_update_measurements)
            measurements = {k: round(float(v), 2) for k, v in self.measurers[model_key].measurements.items()}
        json_dict = {
            'model_type': model_key,
            'gender': self._body_model_gender.selected_text.lower(),
            'preset_id': getattr(self, '_current_preset_id', 'custom'),
            'betas': [round(float(b), 4) for b in self._body_beta_tensor[0].numpy().tolist()],
            'expression': [round(float(e), 4) for e in self._body_exp_tensor[0].numpy().tolist()],
            'measurements': measurements,
        }
        json_path = os.path.splitext(filename)[0] + '.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_dict, f, indent=2, ensure_ascii=False)
        logger.debug(f'Saving JSON params to {json_path}')

    def _on_file_dialog_cancel(self):
        self.window.close_dialog()

    def _on_load_dialog_done(self, filename):
        # Chiamata quando l'utente conferma un file da aprire: chiude il
        # dialogo e delega il caricamento vero e proprio al metodo self.load
        self.window.close_dialog()
        self.load(filename)

    def _on_menu_export(self):
        # Callback di "File > Export Current Image": mostra il dialogo per
        # scegliere dove salvare uno screenshot della vista 3D corrente in PNG
        dlg = gui.FileDialog(gui.FileDialog.SAVE, "Choose file to save",
                             self.window.theme)
        dlg.add_filter(".png", "PNG files (.png)")
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_export_dialog_done)
        self.window.show_dialog(dlg)

    def _on_export_dialog_done(self, filename):
        self.window.close_dialog()
        frame = self._scene.frame
        self.export_image(filename, frame.width, frame.height)

    def _on_menu_quit(self):
        # Callback di "File > Quit": chiude l'intera applicazione
        gui.Application.instance.quit()

    def _on_menu_toggle_settings_panel(self):
        # Callback di "Settings > Model - Lighting - Materials": mostra o
        # nasconde il pannello impostazioni laterale, aggiornando anche il
        # segno di spunta nella voce di menu.
        self._settings_panel.visible = not self._settings_panel.visible
        gui.Application.instance.menubar.set_checked(
            AppWindow.MENU_SHOW_SETTINGS, self._settings_panel.visible)

    def _on_menu_about(self):
        # Show a simple dialog. Although the Dialog is actually a widget, you can
        # treat it similar to a Window for layout and put all the widgets in a
        # layout which you make the only child of the Dialog.
        # (In pratica: costruiamo una piccola finestra di dialogo "About" con
        # un elenco di scorciatoie da tastiera per l'utente, e un bottone OK
        # per chiuderla.)
        em = self.window.theme.font_size
        dlg = gui.Dialog("About")

        # Add the text
        dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
        dlg_layout.add_child(gui.Label("Body Model Visualizer - Help"))
        dlg_layout.add_child(gui.Label("Select joint: Ctrl+left click"))
        dlg_layout.add_child(gui.Label("-- Move selected Joint --"))
        dlg_layout.add_child(gui.Label("Move -x/+x: 1/2"))
        dlg_layout.add_child(gui.Label("Move -y/+y: 3/4"))
        dlg_layout.add_child(gui.Label("Move -z/+z: 5/6"))
        # Add the Ok button. We need to define a callback function to handle
        # the click.
        ok = gui.Button("OK")
        ok.set_on_clicked(self._on_about_ok)

        # We want the Ok button to be an the right side, so we need to add
        # a stretch item to the layout, otherwise the button will be the size
        # of the entire row. A stretch item takes up as much space as it can,
        # which forces the button to be its minimum size.
        h = gui.Horiz()
        h.add_stretch()
        h.add_child(ok)
        h.add_stretch()
        dlg_layout.add_child(h)

        dlg.add_child(dlg_layout)
        self.window.show_dialog(dlg)

    def _on_about_ok(self):
        self.window.close_dialog()

    def add_ground_plane(self):
        # Crea e aggiunge alla scena il pavimento a scacchiera (una serie di
        # piccoli parallelepipedi disposti a griglia, alternati per colore,
        # generati da get_checkerboard_plane in utils.py)
        logger.info('drawing ground plane')
        gp = get_checkerboard_plane(plane_width=2, num_boxes=9)

        for idx, g in enumerate(gp):
            g.compute_vertex_normals()  # Necessario per un'illuminazione corretta di ogni "riquadro" della scacchiera
            self._scene.scene.add_geometry(f"__ground_{idx:04d}__", g, self.settings._materials[Settings.LIT])

    def preload_body_models(self):
        # Carica in memoria (una sola volta, all'avvio) TUTTI i modelli di
        # corpo per tutte le combinazioni modello/genere supportate, e li
        # tiene in cache in PRELOADED_BODY_MODELS. Caricare un modello SMPL
        # da disco è un'operazione relativamente lenta, quindi conviene
        # farlo una volta sola invece che ogni volta che l'utente cambia
        # modello/genere dal menu.
        from smplx import SMPL, SMPLX, MANO, FLAME

        for body_model in AppWindow.BODY_MODEL_NAMES:
            for gender in AppWindow.BODY_MODEL_GENDERS[body_model]:
                logger.info(f'Loading {body_model}-{gender}')
                extra_params = {'gender': gender}
                if body_model in ('SMPLX', 'MANO', 'FLAME'):
                    # Alcuni parametri sono richiesti solo dai modelli con mani/volto:
                    # use_pca=False -> rappresenta la posa delle dita in modo esplicito invece che con componenti PCA compresse
                    # flat_hand_mean=True -> la posa di riposo della mano è "piatta" (dita distese) invece della media statistica
                    # use_face_contour=True -> include i keypoint del contorno del viso
                    extra_params['use_pca'] = False
                    extra_params['flat_hand_mean'] = True
                    extra_params['use_face_contour'] = True
                # "eval(body_model.upper())" trasforma la stringa "SMPL" nella
                # classe SMPL vera e propria (un trucco per scegliere
                # dinamicamente quale classe istanziare in base al nome)
                model = eval(body_model.upper())(f'data/body_models/{body_model.lower()}', **extra_params)
                AppWindow.PRELOADED_BODY_MODELS[f'{body_model.lower()}-{gender.lower()}'] = model
        logger.info(f'Loaded body models {AppWindow.PRELOADED_BODY_MODELS.keys()}')

    # @torch.no_grad()
    def load_body_model(self, body_model='smpl', gender='neutral'):
        # Questo è il metodo "motore" del programma: prende i parametri
        # correnti (betas, espressione, posa) e il modello di corpo
        # scelto/genere, calcola (tramite la rete/formula SMPL già caricata
        # in memoria) i vertici 3D risultanti, costruisce la mesh Open3D e la
        # mostra nella scena. Viene richiamato ogni volta che l'utente
        # cambia un qualunque parametro (beta, posa, espressione, modello...).
        self._scene.scene.remove_geometry("__body_model__")  # Rimuove la mesh precedente prima di disegnare quella nuova

        model = AppWindow.PRELOADED_BODY_MODELS[f'{body_model.lower()}-{gender.lower()}']

        # Copiamo i parametri di posa correnti (per non modificare per
        # sbaglio l'originale) e li "appiattiamo" (reshape) nella forma
        # attesa dal modello: un'unica riga di numeri invece di una
        # sequenza (n_giunti, 3).
        input_params = copy.deepcopy(AppWindow.POSE_PARAMS[body_model])

        for k, v in input_params.items():
            input_params[k] = v.reshape(1, -1)

        # Chiamiamo il modello (è una rete/funzione differenziabile PyTorch):
        # dati i betas (forma), l'espressione e la posa, restituisce i
        # vertici 3D della mesh e le posizioni dei giunti.
        model_output = model(
            betas=self._body_beta_tensor,
            expression=self._body_exp_tensor,
            **input_params,
        )
        verts = model_output.vertices[0].detach().numpy()  # "detach()" stacca il tensore dal grafo dei gradienti prima di convertirlo in numpy
        AppWindow.JOINTS = model_output.joints[0].detach().numpy()
        faces = model.faces  # Le facce (triangoli) sono fisse per ogni modello: non cambiano con la forma/posa

        # ---- Logica dell'ALTEZZA TARGET ----
        # Se l'utente ha impostato un'altezza desiderata (> 0), dobbiamo
        # trovare il valore di beta[0] (il parametro di forma che controlla
        # prevalentemente l'altezza) che produce quell'altezza esatta.
        # Non esiste una formula diretta beta -> altezza, quindi usiamo un
        # metodo numerico iterativo: il METODO DELLE SECANTI. Funziona così:
        # partendo da due stime (b0, b1) del parametro, e le rispettive
        # altezze ottenute (h0, h1), si stima la "pendenza" della funzione
        # altezza(beta) tra i due punti e si estrapola linearmente il valore
        # di beta che azzererebbe la differenza (h - target_h). Si ripete
        # per poche iterazioni, convergendo rapidamente perché la relazione
        # tra beta[0] e altezza è quasi lineare.
        if hasattr(self, '_target_height_val') and self._target_height_val.double_value > 0.0:
            target_h = self._target_height_val.double_value / 100.0  # L'input è in cm, il modello lavora in metri

            def eval_height(b0):
                # Funzione di supporto: dato un valore di beta[0], calcola la
                # mesh risultante e ne misura l'altezza (in metri), poi
                # ripristina il valore originale di beta[0] per non alterare
                # lo stato mentre stiamo "sondando" diversi valori.
                old_b0 = self._body_beta_tensor[0, 0].item()
                self._body_beta_tensor[0, 0] = b0
                mo = model(betas=self._body_beta_tensor, expression=self._body_exp_tensor, **input_params)
                self._body_beta_tensor[0, 0] = old_b0
                v = mo.vertices[0].detach().numpy()
                if 'smplx' in body_model.lower():
                    # Per SMPLX usiamo due vertici "noti" della mesh: uno in
                    # cima alla testa (ht) e la media di due vertici sotto i
                    # piedi (hl), la cui distanza euclidea approssima l'altezza.
                    ht = v[8976]
                    hl = (v[8847] + v[8635]) / 2.0
                    return np.linalg.norm(ht - hl)
                elif 'smpl' in body_model.lower():
                    # Stessa idea ma con gli indici di vertice specifici della topologia SMPL
                    ht = v[412]
                    hl = (v[3458] + v[6858]) / 2.0
                    return np.linalg.norm(ht - hl)
                else:
                    # Per gli altri modelli (senza indici di vertice noti)
                    # approssimiamo l'altezza con l'estensione verticale
                    # della bounding box (differenza tra Y massima e minima)
                    return v[:, 1].max() - v[:, 1].min()

            # Prima stima: il beta[0] attuale, e uno spostato di +0.1 come
            # secondo punto per calcolare la pendenza iniziale
            b0 = self._body_beta_tensor[0, 0].item()
            h0 = eval_height(b0)
            b1 = b0 + 0.1
            h1 = eval_height(b1)

            # Iteriamo il metodo delle secanti per 3 volte: di solito
            # bastano poche iterazioni per avvicinarsi molto al target,
            # dato che la relazione beta[0]->altezza è quasi lineare.
            for _ in range(3):
                if h1 == h0: break  # Evitiamo la divisione per zero se le due altezze coincidono
                # Formula del metodo delle secanti: stima il nuovo beta che
                # azzererebbe l'errore (h1 - target_h), poi lo limitiamo
                # (clip) in un intervallo ragionevole per evitare forme irrealistiche
                b_new = np.clip(b1 - (h1 - target_h) * (b1 - b0) / (h1 - h0), -10.0, 10.0)
                b0, b1 = b1, b_new
                h0, h1 = h1, eval_height(b1)

            # Applichiamo il beta[0] trovato e aggiorniamo sia lo slider
            # corrispondente sia l'etichetta testuale dei betas
            self._body_beta_tensor[0, 0] = b1
            self._body_beta_sliders[0].double_value = b1
            self._body_beta_text.text = f",".join(f'{x:.1f}' for x in self._body_beta_tensor[0].numpy().tolist())

            # Ricalcoliamo la mesh definitiva con il beta[0] ottimizzato
            model_output = model(
                betas=self._body_beta_tensor,
                expression=self._body_exp_tensor,
                **input_params,
            )
            verts = model_output.vertices[0].detach().numpy()
            AppWindow.JOINTS = model_output.joints[0].detach().numpy()
            faces = model.faces

        # Costruiamo la mesh Open3D vera e propria a partire da vertici e facce numpy
        mesh = o3d.geometry.TriangleMesh()

        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.compute_vertex_normals()  # Necessario per l'illuminazione realistica della mesh

        # ---- TEXTURE DELL'AVATAR ----
        # Se l'utente ha scelto una texture (e siamo su SMPL, l'unico modello
        # con template UV ufficiale), assegniamo alla mesh le coordinate UV
        # per-triangolo del template e l'immagine texture. Altrimenti,
        # colore grigio uniforme come prima.
        self._body_texture_img = None  # Immagine texture applicata alla mesh corrente (serve anche all'export OBJ)
        texture_path = self._get_selected_texture_path()
        if texture_path is not None and body_model.lower() == 'smpl':
            tri_uvs = self._get_smpl_triangle_uvs()
            if tri_uvs is not None and len(tri_uvs) == 3 * len(faces):
                mesh.triangle_uvs = o3d.utility.Vector2dVector(tri_uvs)
                self._body_texture_img = o3d.io.read_image(texture_path)
                # textures + triangle_material_ids servono anche per l'export
                # OBJ: così write_triangle_mesh scrive il file .mtl e la texture
                mesh.textures = [self._body_texture_img]
                mesh.triangle_material_ids = o3d.utility.IntVector(
                    np.zeros(len(faces), dtype=np.int32))
            else:
                logger.warning(
                    f'UV template missing or incompatible '
                    f'({AppWindow.TEXTURE_DIR}/{AppWindow.UV_TEMPLATE_FILE}): texture disabled')
        if self._body_texture_img is None:
            mesh.paint_uniform_color([0.5, 0.5, 0.5])  # Colore grigio uniforme di base
        # ipdb.set_trace()
        # Il modello SMPL per default può avere i piedi sotto il livello y=0:
        # calcoliamo di quanto serve traslare verticalmente la mesh (min_y)
        # per farla "appoggiare" esattamente sul pavimento (y=0), e applichiamo
        # la stessa traslazione anche ai giunti, così restano coerenti con la mesh.
        min_y = -mesh.get_min_bound()[1]
        mesh.translate([0, min_y, 0])
        AppWindow.JOINTS += np.array([0, min_y, 0])

        self.current_mesh = mesh

        if self._body_texture_img is not None:
            # Materiale dedicato con la texture come albedo: NON riusiamo
            # self.settings.material perché è condiviso con le altre geometrie
            # della scena (pavimento, overlay...) e verrebbero testurizzate anche loro
            body_material = rendering.MaterialRecord()
            body_material.shader = Settings.LIT
            body_material.albedo_img = self._body_texture_img
            body_material.base_color = [1.0, 1.0, 1.0, 1.0]  # Bianco: la texture non viene scurita
            self._scene.scene.add_geometry("__body_model__", mesh, body_material)
        else:
            self._scene.scene.add_geometry("__body_model__", mesh,
                                           self.settings.material)
        bounds = mesh.get_axis_aligned_bounding_box()
        if AppWindow.CAM_FIRST:
            # Solo al primo caricamento (o dopo un cambio di modello)
            # posizioniamo automaticamente la camera per inquadrare l'intera mesh
            self._scene.setup_camera(60, bounds, bounds.get_center())
            AppWindow.CAM_FIRST = False
        AppWindow.BODY_TRANSL = torch.tensor([[0, min_y, 0]])  # Salviamo la traslazione applicata, utile per l'IK
        self._on_show_joints(self._show_joints.checked)  # Ridisegna i marcatori dei giunti nella nuova posizione

        # Recalculate measurements on mesh load
        # Dopo aver aggiornato la mesh, ricalcoliamo anche le misure
        # antropometriche correnti (altezza, circonferenze, ecc.) e
        # aggiorniamo le etichette nel pannello "Measurements".
        translated_verts = np.asarray(mesh.vertices)
        self.recalculate_and_update_measurements(body_model, translated_verts, AppWindow.JOINTS, gender)

    def load(self, path):
        # Carica un file 3D generico da disco (usato da "File > Open"),
        # diverso da load_body_model perché qui il file può essere sia una
        # mesh triangolare che una semplice nuvola di punti, in vari formati.
        # self._scene.scene.clear_geometry()
        # if self.settings.show_ground:
        #     self.add_ground_plane()

        geometry = None
        # Chiediamo a Open3D di "annusare" il tipo di contenuto del file
        # (mesh con triangoli, nuvola di punti, ecc.) leggendone l'estensione/intestazione
        geometry_type = o3d.io.read_file_geometry_type(path)

        mesh = None
        if geometry_type & o3d.io.CONTAINS_TRIANGLES:
            mesh = o3d.io.read_triangle_mesh(path)
        if mesh is not None:
            if len(mesh.triangles) == 0:
                # Alcuni file "mesh" in realtà non contengono triangoli
                # (es. un file .ply con soli punti): in tal caso lo trattiamo
                # come point cloud più avanti
                print(
                    "[WARNING] Contains 0 triangles, will read as point cloud")
                mesh = None
            else:
                mesh.compute_vertex_normals()
                if len(mesh.vertex_colors) == 0:
                    mesh.paint_uniform_color([1, 1, 1])
                geometry = mesh
            # Make sure the mesh has texture coordinates
            # (Alcuni shader richiedono coordinate UV anche se non c'è una
            # texture vera e propria: qui le riempiamo con zeri "fittizi"
            # per evitare errori di rendering.)
            if not mesh.has_triangle_uvs():
                uv = np.array([[0.0, 0.0]] * (3 * len(mesh.triangles)))
                mesh.triangle_uvs = o3d.utility.Vector2dVector(uv)
        else:
            print("[Info]", path, "appears to be a point cloud")

        if geometry is None:
            # Se non siamo riusciti a leggere una mesh valida, proviamo a
            # interpretare il file come una nuvola di punti (point cloud)
            cloud = None
            try:
                cloud = o3d.io.read_point_cloud(path)
            except Exception:
                pass
            if cloud is not None:
                print("[Info] Successfully read", path)
                if not cloud.has_normals():
                    cloud.estimate_normals()  # Calcola le normali stimandole dai punti vicini, se non presenti nel file
                cloud.normalize_normals()
                geometry = cloud
            else:
                print("[WARNING] Failed to read points", path)

        if geometry is not None:
            try:
                # Aggiungiamo la geometria caricata alla scena (sostituendo
                # quella con lo stesso nome "__model__" se già presente) e
                # posizioniamo la camera per inquadrarla per intero
                self._scene.scene.add_geometry("__model__", geometry,
                                               self.settings.material)
                bounds = geometry.get_axis_aligned_bounding_box()
                self._scene.setup_camera(60, bounds, bounds.get_center())
            except Exception as e:
                print(e)

    def export_image(self, path, width, height):
        # Esporta la vista 3D corrente come immagine (PNG o JPG) sul percorso indicato

        def on_image(image):
            # Callback asincrona chiamata da Open3D quando l'immagine
            # renderizzata della scena è pronta
            img = image

            quality = 9  # png
            if path.endswith(".jpg"):
                quality = 100
            o3d.io.write_image(path, img, quality)

        self._scene.scene.scene.render_to_image(on_image)

    def recalculate_and_update_measurements(self, body_model, verts, joints, gender):
        # Calcola le misure antropometriche (altezza, circonferenze, ecc.)
        # della mesh corrente usando la classe MeasureBody (dal sottoprogetto
        # SMPL-Anthropometry-master) e aggiorna le etichette "Current
        # Measurements" nella GUI. Viene chiamato ogni volta che la mesh
        # cambia (nuovo modello, nuovi betas, nuova posa...).
        model_key = body_model.lower()
        if model_key not in ('smpl', 'smplx'):
            # Il calcolo delle misure è implementato solo per SMPL/SMPLX:
            # per gli altri modelli (MANO, FLAME) mostriamo "N/A" (non disponibile)
            if hasattr(self, 'current_measure_labels'):
                for k in self.current_measure_labels.keys():
                    self.current_measure_labels[k].text = "N/A"
            return

        if not hasattr(self, 'measurers'):
            # Creiamo (una sola volta, la prima volta che serve) gli oggetti
            # MeasureBody per SMPL e SMPLX, e li teniamo in cache in self.measurers
            self.measurers = {
                'smpl': MeasureBody('smpl'),
                'smplx': MeasureBody('smplx')
            }

        measurer = self.measurers[model_key]
        # Passiamo al misuratore i vertici e i giunti della mesh corrente e il genere
        measurer.verts = verts
        measurer.joints = joints
        measurer.gender = gender.upper()

        measurer.measurements = {}
        # Calcola TUTTE le misure possibili supportate da MeasureBody (altezza,
        # circonferenze di vita/petto/fianchi/collo/testa, lunghezze di
        # braccia/gambe, larghezza spalle, ecc.)
        measurer.measure(measurer.all_possible_measurements)

        # Aggiorniamo ogni etichetta della GUI con il valore misurato
        # corrispondente (in cm, con una cifra decimale), o "N/A" se quella
        # specifica misura non è stata calcolata.
        for k, label in self.current_measure_labels.items():
            if k in measurer.measurements:
                val = measurer.measurements[k]
                label.text = f"{val:.1f} cm"
            else:
                label.text = "N/A"

        # Aggiorniamo anche l'eventuale overlay grafico (linea/circonferenza) visibile
        self._update_overlay()

    def _on_visible_overlay_changed(self, name, index):
        # Callback del menu a tendina "Visible Overlay": basta ridisegnare l'overlay
        self._update_overlay()

    def _update_overlay(self):
        # Disegna sopra la mesh un elemento grafico (linea o poligono di
        # circonferenza) che rappresenta VISIVAMENTE dove viene presa la
        # misura selezionata nel menu "Visible Overlay". Utile per capire
        # a colpo d'occhio cosa significa, ad esempio, "waist circumference".

        # Prima rimuoviamo sempre gli eventuali overlay disegnati in precedenza
        if self._scene.scene.has_geometry("__overlay__"):
            self._scene.scene.remove_geometry("__overlay__")
        if self._scene.scene.has_geometry("__overlay_endpoint1__"):
            self._scene.scene.remove_geometry("__overlay_endpoint1__")
        if self._scene.scene.has_geometry("__overlay_endpoint2__"):
            self._scene.scene.remove_geometry("__overlay_endpoint2__")

        if not hasattr(self, '_visible_overlay'):
            return

        selected = self._visible_overlay.selected_text
        if selected == "None":
            # L'utente non vuole vedere nessun overlay
            return

        model_key = self._body_model.selected_text.lower()
        if model_key not in ('smpl', 'smplx') or not hasattr(self, 'measurers'):
            return

        measurer = self.measurers[model_key]
        if selected not in measurer.measurements:
            # La misura scelta non è stata calcolata per il modello corrente
            return

        if not hasattr(measurer, 'measurement_geometries') or selected not in measurer.measurement_geometries:
            # Il misuratore non ha salvato la geometria (punti/segmenti) di
            # questa specifica misura, quindi non possiamo disegnarla
            return

        # measurement_geometries contiene, per ciascuna misura, il TIPO di
        # geometria da disegnare ('length' per una lunghezza tra due punti,
        # 'circumference' per una circonferenza) seguito dai dati geometrici
        # necessari (i due punti, oppure la lista di segmenti della sezione).
        geom_type, *geom_data = measurer.measurement_geometries[selected]

        # Materiale rosso per la linea/overlay principale, spesso (line_width)
        # per renderlo ben visibile sopra la mesh
        mat = rendering.MaterialRecord()
        mat.base_color = [1.0, 0.1, 0.1, 1.0]  # Red
        mat.shader = "defaultUnlit"
        mat.line_width = 8.0

        if geom_type == 'length':
            # Caso "lunghezza": disegniamo un semplice segmento tra due punti
            # (es. per la lunghezza del braccio, dalla spalla al polso)
            p1, p2 = geom_data

            lineset = o3d.geometry.LineSet()
            lineset.points = o3d.utility.Vector3dVector([p1, p2])
            lineset.lines = o3d.utility.Vector2iVector([[0, 1]])

            self._scene.scene.add_geometry("__overlay__", lineset, mat)

            # Aggiungiamo anche due piccole sfere verdi ai due estremi, per
            # evidenziare chiaramente dove iniziano/finiscono la misura
            sp_mat = rendering.MaterialRecord()
            sp_mat.base_color = [0.1, 0.9, 0.1, 1.0]  # Green endpoints
            sp_mat.shader = "defaultLit"

            sp1 = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
            sp1.compute_vertex_normals()
            sp1.translate(p1)
            self._scene.scene.add_geometry("__overlay_endpoint1__", sp1, sp_mat)

            sp2 = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
            sp2.compute_vertex_normals()
            sp2.translate(p2)
            self._scene.scene.add_geometry("__overlay_endpoint2__", sp2, sp_mat)

        elif geom_type == 'circumference':
            # Caso "circonferenza": la misura è ottenuta "affettando" la mesh
            # con un piano (es. all'altezza della vita) e prendendo il
            # perimetro convesso (hull) dell'intersezione. slice_segments_hull
            # è quindi una sequenza di piccoli segmenti che, uniti, formano
            # l'intera circonferenza da disegnare.
            slice_segments_hull = geom_data[0]

            num_segments = slice_segments_hull.shape[0]
            points = []
            lines = []
            for idx in range(num_segments):
                seg = slice_segments_hull[idx]
                points.append(seg[0])
                points.append(seg[1])
                lines.append([idx * 2, idx * 2 + 1])

            lineset = o3d.geometry.LineSet()
            lineset.points = o3d.utility.Vector3dVector(points)
            lineset.lines = o3d.utility.Vector2iVector(lines)

            self._scene.scene.add_geometry("__overlay__", lineset, mat)

    def _on_fit_measurements(self):
        # Callback del bottone "Fit Avatar to Targets": è il punto di
        # ingresso della funzionalità di "fitting" automatico. Raccoglie
        # dalla GUI tutte le misure target inserite dall'utente (ignorando
        # quelle lasciate a 0, che significano "nessun vincolo"), poi avvia
        # l'ottimizzazione vera e propria.
        targets = {}
        for k, num_edit in self.target_inputs.items():
            if num_edit.double_value > 0.0:
                targets[k] = num_edit.double_value

        if not targets:
            # Se l'utente non ha inserito nessuna misura valida, avvisiamo e usciamo
            self._update_label("Please specify at least one target measurement > 0 cm.")
            return

        body_model_name = self._body_model.selected_text
        gender_name = self._body_model_gender.selected_text

        self._update_label("Running optimization... Please wait.")
        # Disabilitiamo il bottone e mostriamo lo stato, così è chiaro che
        # l'ottimizzazione è in corso (può bloccare la GUI per qualche secondo)
        self._fit_measurements_btn.enabled = False
        self._fit_status.text = "Fitting in progress..."
        # Schedulare l'ottimizzazione con "post_to_main_thread" (anche se qui
        # veniamo già chiamati dal thread principale) è un modo sicuro per
        # assicurarsi che il messaggio "Running optimization..." venga
        # effettivamente disegnato PRIMA che inizi il calcolo (che può
        # richiedere qualche istante e bloccare l'interfaccia nel frattempo).
        gui.Application.instance.post_to_main_thread(
            self.window, lambda: self._run_optimization_fit(body_model_name, gender_name, targets)
        )

    def _run_optimization_fit(self, body_model_name, gender_name, targets):
        # Cuore dell'algoritmo di "fitting": cerca i 10 parametri di forma
        # (betas) che fanno sì che le misure del corpo generato si avvicinino
        # il più possibile alle misure target inserite dall'utente.
        # L'ottimizzazione vera e propria (scipy L-BFGS-B con derivate
        # numeriche) vive nella funzione condivisa fit_betas_to_measurements
        # del modulo fitting.py, la stessa usata da build_presets.py per
        # precalcolare le betas dei preset di avatar.
        model = AppWindow.PRELOADED_BODY_MODELS[f'{body_model_name.lower()}-{gender_name.lower()}']

        measurer = self.measurers[body_model_name.lower()]
        initial_betas = self._body_beta_tensor[0].numpy().copy()  # Punto di partenza dell'ottimizzazione: i betas attuali

        # La posa corrente resta fissa durante l'ottimizzazione:
        # modifichiamo solo la FORMA, non la posa
        optimized_betas, res = fit_betas_to_measurements(
            model, measurer, targets,
            initial_betas=initial_betas,
            expression=self._body_exp_tensor,
            pose_params=AppWindow.POSE_PARAMS[body_model_name],
        )

        self._mark_custom_preset()  # Il fit produce una forma personalizzata: il preset non è più quello applicato
        self._body_beta_tensor[0] = torch.from_numpy(optimized_betas).float()

        # Aggiorniamo gli slider e l'etichetta testuale con i betas trovati
        for idx in range(10):
            self._body_beta_sliders[idx].double_value = float(optimized_betas[idx])

        self._body_beta_text.text = f",".join(f'{x:.1f}' for x in self._body_beta_tensor[0].numpy().tolist())

        # Ricarichiamo la mesh definitiva con i betas ottimizzati, e
        # mostriamo all'utente il valore finale della funzione di costo
        # (più è basso, migliore è stato il fitting)
        self.load_body_model(body_model_name, gender=gender_name)

        # load_body_model ha appena ricalcolato TUTTE le misure correnti nel
        # measurer: le confrontiamo con i target per mostrare l'errore
        # residuo massimo, un feedback più concreto della semplice loss.
        errors = [abs(measurer.measurements[k] - v)
                  for k, v in targets.items() if k in measurer.measurements]
        max_err = max(errors) if errors else 0.0
        self._fit_status.text = f"Fit completed - max error {max_err:.1f} cm"
        self._fit_measurements_btn.enabled = True

        self._update_label(f"Avatar fit successfully! Loss: {res.fun:.4f}")

    def _on_export_obj_dialog(self):
        # Callback di "File > Export 3D Mesh (OBJ)": mostra il dialogo per
        # scegliere dove salvare la mesh corrente in formato Wavefront OBJ
        # (un formato di file 3D testuale molto diffuso e supportato da
        # praticamente tutti i software di modellazione/stampa 3D).
        dlg = gui.FileDialog(gui.FileDialog.SAVE, "Choose file to save OBJ",
                             self.window.theme)
        dlg.add_filter(".obj", "Wavefront OBJ files (.obj)")
        dlg.set_on_cancel(self._on_save_dialog_cancel)
        dlg.set_on_done(self._on_export_obj_dialog_done)
        self.window.show_dialog(dlg)

    def _on_export_obj_dialog_done(self, filename):
        # Chiamata quando l'utente conferma il percorso: scrive su disco
        # la mesh attualmente visualizzata (self.current_mesh, salvata in
        # load_body_model) nel formato OBJ
        self.window.close_dialog()
        if hasattr(self, 'current_mesh') and self.current_mesh is not None:
            logger.debug(f'Exporting OBJ mesh to {filename}')
            o3d.io.write_triangle_mesh(filename, self.current_mesh)
            self._update_label(f"Mesh exported to {filename}")
        else:
            logger.warning("No mesh to export")


# ============================================================================
# FUNZIONE main() E PUNTO DI INGRESSO DEL PROGRAMMA
# ============================================================================
def main(args):
    # Funzione principale che avvia l'intera applicazione: inizializza la
    # libreria grafica di Open3D, crea la finestra e avvia il "loop degli
    # eventi" (event loop), cioè il ciclo che tiene in vita la GUI e
    # reagisce a ogni interazione dell'utente (click, movimenti, ecc.)
    # finché la finestra non viene chiusa.
    if args.web:
        # Se l'utente ha passato l'opzione --web, abilitiamo la
        # visualizzazione via browser (WebRTC) invece della finestra nativa
        logger.info('Initializing web visualization')
        o3d.visualization.webrtc_server.enable_webrtc()

    # We need to initalize the application, which finds the necessary shaders
    # for rendering and prepares the cross-platform window abstraction.
    gui.Application.instance.initialize()

    w = AppWindow(1920, 1080)  # Creiamo la finestra principale con risoluzione Full HD+ (1920x1080)

    # Run the event loop. This will not return until the last window is closed.
    gui.Application.instance.run()


# Questo blocco viene eseguito SOLO se il file viene lanciato direttamente
# (es. "python main.py"), non se viene importato come modulo da un altro file.
if __name__ == "__main__":
    # Configuriamo gli argomenti accettati da riga di comando: qui c'è solo
    # "--web", un flag booleano (True se presente, False altrimenti) per
    # scegliere la modalità di visualizzazione via browser.
    parser = argparse.ArgumentParser()
    parser.add_argument('--web', action='store_true', help='Enable web visualization')

    args = parser.parse_args()
    main(args)
