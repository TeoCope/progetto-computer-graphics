
# Import di tipi usati solo per l'annotazione dei parametri (List, Dict),
# non incidono sull'esecuzione ma aiutano a capire cosa si aspetta ogni funzione.
from typing import List, Dict
# numpy: libreria per il calcolo numerico su array/vettori/matrici (usata per le coordinate 3D dei vertici).
import numpy as np
# trimesh: libreria per lavorare con mesh 3D (triangoli, tagli con un piano, convex hull, ecc.).
import trimesh
# torch (PyTorch): libreria per tensori, usata perché i modelli SMPL/SMPLX lavorano con tensori.
import torch
# smplx: libreria che fornisce i modelli parametrici del corpo umano SMPL e SMPLX.
import smplx
# pprint: funzione per stampare a schermo dizionari/liste in modo leggibile ("pretty print").
from pprint import pprint
# os: funzioni di sistema operativo, qui usato per costruire percorsi di file in modo portabile.
import os
# argparse: libreria per leggere argomenti passati da riga di comando (usata nel blocco main in fondo).
import argparse

# Import "wildcard" (con *) da moduli dello stesso progetto: importano tutte le costanti/classi
# pubbliche definite in quei file, così si possono usare direttamente senza prefisso di modulo.
# Contiene le definizioni delle misure (lunghezze, circonferenze) per SMPL e SMPLX.
from measurement_definitions import *
# Funzioni di utilità antropometrica: es. filtro dei segmenti per parte del corpo, convex hull, caricamento segmentazione facce.
from anthro_utils import *
# Classe Visualizer per disegnare la mesh 3D con misure, landmark e giunti sovrapposti.
from visualize import Visualizer
# Definizioni degli indici dei landmark (punti di riferimento anatomici) sulla mesh SMPL/SMPLX.
from landmark_definitions import *
# Definizioni e mapping dei giunti (joint) del corpo per SMPL/SMPLX.
from joint_definitions import *



# Funzione che applica dei parametri di forma (shape) a un modello di corpo già creato,
# restituendo il modello "deformato" secondo quella forma (con vertici calcolati).
def set_shape(model, shape_coefs):
    '''
    Set shape of body model.
    :param model: smplx body model
    :param shape_coefs: torch.tensor dim (10,)

    Return
    shaped smplx body model
    '''
    # Converte i coefficienti di forma (beta) in float32, tipo richiesto dai tensori PyTorch usati da smplx.
    shape_coefs = shape_coefs.to(torch.float32)
    # Chiama il modello come una funzione (forward pass): passandogli i beta ottiene in uscita
    # un oggetto che contiene, tra le altre cose, i vertici (verts) e i giunti (joints) del corpo con quella forma.
    return model(betas=shape_coefs, return_verts=True)

# Funzione factory che crea/carica un modello di corpo parametrico (SMPL, SMPLX, ecc.)
# usando la libreria smplx, a partire dal tipo di modello e dai parametri di configurazione.
def create_model(model_type, model_root, gender, num_betas=10, num_thetas=24):
    '''
    Create SMPL/SMPLX/etc. body model
    :param model_type: str of model type: smpl, smplx, etc.
    :param model_root: str of location where there are smpl/smplx/etc. folders with .pkl models
                        (clumsy definition in smplx package)
    :param gender: str of gender: MALE or FEMALE or NEUTRAL
    :param num_betas: int of number of shape coefficients
                      requires the model with num_coefs in model_root
    :param num_thetas: int of number of pose coefficients
    
    Return:
    :param smplx body model (SMPL, SMPLX, etc.)
    '''
    
    #body_pose = torch.zeros((1, (num_thetas-1) * 3))
    
    # Sceglie l'estensione del file del modello in base al tipo:
    # SMPLX viene salvato/caricato come file .npz (formato numpy compresso),
    # mentre SMPL (e altri modelli) usano il formato .pkl (pickle Python).
    ext_val = 'npz' if model_type.lower() == 'smplx' else 'pkl'
    # smplx.create() è la funzione della libreria smplx che istanzia il modello del corpo,
    # cercando i file dei pesi/parametri dentro model_root/model_type con l'estensione scelta sopra.
    return smplx.create(model_path=model_root,
                        model_type=model_type,
                        gender=gender, 
                        use_face_contour=False,
                        num_betas=num_betas,
                        #body_pose=body_pose,
                        ext=ext_val)



class Measurer():
    '''
    Measure a parametric body model defined either.
    Parent class for Measure{SMPL,SMPLX,..}.

    All the measurements are expressed in cm.
    '''

    def __init__(self):
        # Vertici della mesh (array Nx3 di coordinate 3D), inizialmente non definiti (None)
        # finché non viene chiamato from_verts() o from_body_model().
        self.verts = None
        # Facce (triangoli) della mesh: array di indici che indicano quali 3 vertici formano ogni triangolo.
        self.faces = None
        # Posizioni 3D dei giunti (joints) del corpo (es. spalle, gomiti, ginocchia).
        self.joints = None
        # Genere del corpo misurato (MALE/FEMALE/NEUTRAL), impostato quando si costruisce il modello.
        self.gender = None

        # Dizionario che conterrà i risultati delle misure: {nome_misura: valore_in_cm}.
        self.measurements = {}
        # Dizionario delle misure "riscalate" rispetto a un'altezza di riferimento (vedi height_normalize_measurements).
        self.height_normalized_measurements = {}
        # Dizionario delle misure con etichette personalizzate al posto dei nomi originali (vedi label_measurements).
        self.labeled_measurements = {}
        # Come sopra, ma con i valori riscalati per altezza.
        self.height_normalized_labeled_measurements = {}
        # Mappa che associa ogni etichetta (label) al nome di misura originale corrispondente.
        self.labels2names = {}

    def from_verts(self):
        # Metodo "placeholder": nella classe base non fa nulla, viene ridefinito (override)
        # nelle sottoclassi MeasureSMPL/MeasureSMPLX per costruire il modello a partire dai vertici.
        pass

    def from_body_model(self):
        # Come sopra: placeholder ridefinito nelle sottoclassi per costruire il modello
        # a partire da genere e parametri di forma (shape).
        pass

    def measure(self, 
                measurement_names: List[str]
                ):
        '''
        Measure the given measurement names from measurement_names list
        :param measurement_names - list of strings of defined measurements
                                    to measure from MeasurementDefinitions class
        '''

        # Dispatcher: per ogni nome di misura richiesto, decide se calcolarla come
        # "lunghezza" (distanza fra due punti) o come "circonferenza" (taglio della mesh + convex hull).
        for m_name in measurement_names:
            if m_name not in self.all_possible_measurements:
                # Se il nome misura non è tra quelle definite, stampa un avviso.
                # NOTA: "pass" qui non interrompe il ciclo, quindi il codice prosegue comunque sotto.
                print(f"Measurement {m_name} not defined.")
                pass

            if m_name in self.measurements:
                # Se la misura è già stata calcolata in precedenza, non fa nulla di esplicito qui
                # (il "pass" non impedisce il ricalcolo più sotto).
                pass

            if self.measurement_types[m_name] == MeasurementType().LENGTH:

                # Tipo "lunghezza": calcola la distanza euclidea tra due landmark (punti di riferimento).
                value = self.measure_length(m_name)
                self.measurements[m_name] = value

            elif self.measurement_types[m_name] == MeasurementType().CIRCUMFERENCE:

                # Tipo "circonferenza": taglia la mesh con un piano e misura il perimetro del contorno risultante.
                value = self.measure_circumference(m_name)
                self.measurements[m_name] = value
    
            else:
                # Tipo di misura non riconosciuto (né lunghezza né circonferenza).
                print(f"Measurement {m_name} not defined")

    def measure_length(self, measurement_name: str):
        '''
        Measure distance between 2 landmarks
        :param measurement_name: str - defined in MeasurementDefinitions

        Returns
        :float of measurement in cm
        '''

        # Recupera la coppia di indici di landmark associata a questa misura (es. i due punti che definiscono un'altezza).
        measurement_landmarks_inds = self.length_definitions[measurement_name]

        # Lista in cui accumuleremo le coordinate 3D dei due punti (landmark) da usare per calcolare la distanza.
        landmark_points = []
        # Una lunghezza è sempre definita da esattamente 2 landmark: li processiamo uno alla volta (i=0, i=1).
        for i in range(2):
            # Controlla se il landmark è specificato come indice singolo o come coppia di indici (tuple).
            if isinstance(measurement_landmarks_inds[i],tuple):
                # if touple of indices for landmark, take their average
                lm = (self.verts[measurement_landmarks_inds[i][0]] + 
                          self.verts[measurement_landmarks_inds[i][1]]) / 2
            # Caso standard: il landmark è un singolo indice di vertice sulla mesh.
            else:
                lm = self.verts[measurement_landmarks_inds[i]]
            
            # Aggiunge il punto (coordinate x,y,z) alla lista dei due punti che definiscono la lunghezza.
            landmark_points.append(lm)

        # Impila i due punti in un array numpy di forma (2,3) e aggiunge una dimensione extra davanti (diventa (1,2,3)),
        # perché _get_dist si aspetta un array con forma (N,2,3): N misure, ciascuna con 2 punti da 3 coordinate.
        landmark_points = np.vstack(landmark_points)[None,...]

        # Pattern di "caching lazy": se l'attributo measurement_geometries non esiste ancora sull'oggetto,
        # lo crea come dizionario vuoto la prima volta che serve (verrà popolato con le geometrie da disegnare nel visualizzatore).
        if not hasattr(self, 'measurement_geometries'):
            self.measurement_geometries = {}
        # Salva le coordinate dei due punti estremi di questa misura, etichettati come 'length',
        # cosi' il visualizzatore 3D (Visualizer) potra' in seguito disegnare un segmento tra questi due punti come overlay.
        self.measurement_geometries[measurement_name] = ('length', landmark_points[0, 0], landmark_points[0, 1])

        # Calcola e restituisce la distanza euclidea (in cm) tra i due punti trovati.
        return self._get_dist(landmark_points)

    @staticmethod
    def _get_dist(verts: np.ndarray) -> float:
        '''
        The Euclidean distance between vertices.
        The distance is found as the sum of each pair i 
        of 3D vertices (i,0,:) and (i,1,:) 
        :param verts: np.ndarray (N,2,3) - vertices used 
                        to find distances
        
        Returns:
        :param dist: float, sumed distances between vertices
        '''

        # Per ciascuna coppia di punti (verts[:,0] e verts[:,1]) calcola la distanza euclidea in 3D:
        # np.linalg.norm calcola la norma (lunghezza) del vettore differenza (punto1 - punto0) lungo l'asse delle coordinate (axis=1).
        verts_distances = np.linalg.norm(verts[:, 1] - verts[:, 0],axis=1)
        # Somma tutte le distanze parziali (utile quando una misura e' composta da piu' segmenti consecutivi, come nel caso delle circonferenze).
        distance = np.sum(verts_distances)
        # I modelli SMPL/SMPLX esprimono le coordinate in metri: moltiplicare per 100 converte il risultato in centimetri.
        distance_cm = distance * 100 # convert to cm
        return distance_cm
    
    def measure_circumference(self, 
                              measurement_name: str, 
                              ):
        '''
        Measure circumferences. Circumferences are defined with 
        landmarks and joints - the measurement is found by cutting the 
        SMPL model with the  plane defined by a point (landmark point) and 
        normal (vector connecting the two joints).
        :param measurement_name: str - measurement name

        Return
        float of measurement value in cm
        '''

        # Recupera la definizione della circonferenza: contiene i landmark che indicano dove deve passare il piano di taglio
        # e i due giunti che definiscono la direzione (normale) del piano.
        measurement_definition = self.circumf_definitions[measurement_name]
        # Nomi dei landmark da usare per calcolare il punto per cui deve passare il piano di taglio.
        circumf_landmarks = measurement_definition["LANDMARKS"]
        # Converte i nomi dei landmark nei corrispondenti indici di vertice sulla mesh.
        circumf_landmark_indices = [self.landmarks[l_name] for l_name in circumf_landmarks]
        # Recupera i nomi dei due giunti la cui congiungente definisce la direzione perpendicolare al piano di taglio
        # (es. per la circonferenza del torace, i giunti spalla sinistra/destra danno la direzione del taglio).
        circumf_n1, circumf_n2 = self.circumf_definitions[measurement_name]["JOINTS"]
        # Converte i nomi dei due giunti nei corrispondenti indici numerici in self.joints.
        circumf_n1, circumf_n2 = self.joint2ind[circumf_n1], self.joint2ind[circumf_n2]
        
        # Il punto per cui passa il piano di taglio e' la media delle posizioni dei landmark scelti
        # (baricentro dei punti di riferimento).
        plane_origin = np.mean(self.verts[circumf_landmark_indices,:],axis=0)
        # Il vettore normale al piano di taglio e' la differenza tra le posizioni dei due giunti:
        # cosi' si taglia la mesh perpendicolarmente alla direzione che va da un giunto all'altro.
        plane_normal = self.joints[circumf_n1,:] - self.joints[circumf_n2,:]

        # Costruisce un oggetto mesh trimesh a partire dai vertici e dalle facce del corpo,
        # necessario per usare le funzioni geometriche (taglio con piano, ecc.) della libreria trimesh.
        mesh = trimesh.Trimesh(vertices=self.verts, faces=self.faces)

        # new version            
        # trimesh.intersections.mesh_plane taglia la mesh con un piano (definito da plane_origin e plane_normal)
        # e restituisce:
        #  - slice_segments: array (N,2,3) con gli N segmenti 3D dell'intersezione (ogni segmento = 2 punti che formano un pezzetto del contorno di taglio);
        #  - sliced_faces (perche' return_faces=True): array (N,) con l'indice della faccia originale da cui proviene ciascun segmento,
        #    utile per sapere a quale parte del corpo appartiene ogni pezzo del taglio.
        slice_segments, sliced_faces = trimesh.intersections.mesh_plane(mesh, 
                                plane_normal=plane_normal, 
                                plane_origin=plane_origin, 
                                return_faces=True) # (N, 2, 3), (N,)
        
        # La mesh puo' essere tagliata dal piano anche in punti lontani dalla parte del corpo che interessa
        # (es. un piano per il punto vita potrebbe intersecare anche un braccio abbassato): questa funzione filtra
        # i segmenti tenendo solo quelli che appartengono alla parte del corpo giusta per questa specifica misura.
        slice_segments = filter_body_part_slices(slice_segments,
                                                 sliced_faces,
                                                 measurement_name,
                                                 self.circumf_2_bodypart,
                                                 self.face_segmentation)
        
        # I segmenti filtrati formano un contorno nello spazio 3D: calcola il convex hull (inviluppo convesso)
        # di questi punti per ottenere un poligono chiuso che approssima la circonferenza da misurare.
        slice_segments_hull = convex_hull_from_3D_points(slice_segments)

        # Come in measure_length: crea il dizionario di cache delle geometrie se non esiste ancora.
        if not hasattr(self, 'measurement_geometries'):
            self.measurement_geometries = {}
        # Salva i punti del contorno (etichettati come 'circumference') cosi' il Visualizer potra' disegnare
        # il perimetro della circonferenza come overlay sulla mesh 3D.
        self.measurement_geometries[measurement_name] = ('circumference', slice_segments_hull)

        # _get_dist somma le distanze tra punti consecutivi del contorno, ottenendo il perimetro (in cm) della circonferenza.
        return self._get_dist(slice_segments_hull)

    def height_normalize_measurements(self, new_height: float):
        ''' 
        Scale all measurements so that the height measurement gets
        the value of new_height:
        new_measurement = (old_measurement / old_height) * new_height
        NOTE the measurements and body model remain unchanged, a new
        dictionary height_normalized_measurements is created.
        
        Input:
        :param new_height: float, the newly defined height.

        Return:
        self.height_normalized_measurements: dict of 
                {measurement:value} pairs with 
                height measurement = new_height, and other measurements
                scaled accordingly
        '''
        # Procede solo se sono gia' state calcolate delle misure (altrimenti non c'e' nulla da normalizzare).
        if self.measurements != {}:
            # Usa l'altezza attualmente misurata come riferimento per calcolare il fattore di scala.
            old_height = self.measurements["height"]
            # Riscala ogni misura in proporzione: se il rapporto tra nuova e vecchia altezza e' k,
            # tutte le altre misure vengono moltiplicate per lo stesso k (si assume una scala uniforme del corpo).
            for m_name, m_value in self.measurements.items():
                norm_value = (m_value / old_height) * new_height
                self.height_normalized_measurements[m_name] = norm_value

            # Se esistono anche misure con etichette personalizzate, applica la stessa normalizzazione anche a quelle.
            if self.labeled_measurements != {}:
                for m_name, m_value in self.labeled_measurements.items():
                    norm_value = (m_value / old_height) * new_height
                    self.height_normalized_labeled_measurements[m_name] = norm_value

    def label_measurements(self,set_measurement_labels: Dict[str, str]):
        '''
        Create labeled_measurements dictionary with "label: x cm" structure
        for each given measurement.
        NOTE: This overwrites any prior labeling!
        
        :param set_measurement_labels: dict of labels and measurement names
                                        (example. {"A": "head_circumference"})
        '''

        # Avvisa se stiamo per sovrascrivere delle etichette gia' assegnate in precedenza.
        if self.labeled_measurements != {}:
            print("Overwriting old labels")

        # Ripulisce i dizionari di etichette per ricominciare da zero (come indicato nella docstring, sovrascrive tutto).
        self.labeled_measurements = {}
        self.labels2names = {}

        # Per ogni coppia (etichetta, nome_misura) richiesta dall'utente...
        for set_label, set_name in set_measurement_labels.items():
            
            # Controlla che il nome di misura richiesto sia tra quelle effettivamente definite.
            if set_name not in self.all_possible_measurements:
                print(f"Measurement {set_name} not defined.")
                pass

            # Se la misura non e' ancora stata calcolata, la calcola ora (chiamando measure()).
            if set_name not in self.measurements.keys():
                self.measure([set_name])

            # Associa l'etichetta al valore della misura e memorizza anche a quale nome di misura originale corrisponde.
            self.labeled_measurements[set_label] = self.measurements[set_name]
            self.labels2names[set_label] = set_name

    def visualize(self,
                 measurement_names: List[str] = [], 
                 landmark_names: List[str] = [],
                 title="Measurement visualization",
                 visualize_body: bool = True,
                 visualize_landmarks: bool = True,
                 visualize_joints: bool = True,
                 visualize_measurements: bool=True):

        # TODO: create default model if not defined
        # if self.verts is None:
        #     print("Model has not been defined. \
        #           Visualizing on default male model")
        #     model = create_model(self.smpl_path, "MALE", num_coefs=10)
        #     shape = torch.zeros((1, 10), dtype=torch.float32)
        #     model_output = set_shape(model, shape)
            
        #     verts = model_output.vertices.detach().cpu().numpy().squeeze()
        #     faces = model.faces.squeeze()
        # else:
        #     verts = self.verts
        #     faces = self.faces 

        # Se l'utente non specifica quali misure visualizzare, mostra tutte quelle disponibili.
        if measurement_names == []:
            measurement_names = self.all_possible_measurements

        # Allo stesso modo, se non sono specificati i landmark da mostrare, mostra tutti quelli definiti.
        if landmark_names == []:
            landmark_names = list(self.landmarks.keys())

        # Crea un oggetto Visualizer passandogli tutti i dati necessari (mesh, giunti, landmark, misure e le loro definizioni)
        # per poter disegnare la scena 3D interattiva con gli overlay richiesti.
        vizz = Visualizer(verts=self.verts,
                        faces=self.faces,
                        joints=self.joints,
                        landmarks=self.landmarks,
                        measurements=self.measurements,
                        measurement_types=self.measurement_types,
                        length_definitions=self.length_definitions,
                        circumf_definitions=self.circumf_definitions,
                        joint2ind=self.joint2ind,
                        circumf_2_bodypart=self.circumf_2_bodypart,
                        face_segmentation=self.face_segmentation,
                        visualize_body=visualize_body,
                        visualize_landmarks=visualize_landmarks,
                        visualize_joints=visualize_joints,
                        visualize_measurements=visualize_measurements,
                        title=title
                        )
        
        # Avvia effettivamente la finestra di visualizzazione 3D con le misure e i landmark scelti.
        vizz.visualize(measurement_names=measurement_names,
                       landmark_names=landmark_names,
                       title=title)


class MeasureSMPL(Measurer):
    '''
    Measure the SMPL model defined either by the shape parameters or
    by its 6890 vertices. 

    All the measurements are expressed in cm.
    '''

    # Costruttore: qui vengono impostati tutti i dati specifici del modello SMPL
    # (percorsi dei file, landmark, definizioni delle misure, mapping dei giunti).
    def __init__(self):
        
        # Richiama il costruttore della classe base Measurer per inizializzare gli attributi comuni
        # (verts, joints, dizionari delle misure, ecc. impostati a valori vuoti/None).
        super().__init__()

        # Nome del tipo di modello: usato per scegliere l'estensione del file, il joint regressor, ecc.
        self.model_type = "smpl"
        # Cartella radice dove sono salvati i file dei modelli di corpo (SMPL, SMPLX, ...).
        self.body_model_root = "data/body_models"
        # Costruisce il percorso completo alla sottocartella specifica del modello SMPL (es. data/body_models/smpl).
        self.body_model_path = os.path.join(self.body_model_root, 
                                            self.model_type)

        # Carica il modello SMPL (formato file .pkl, tipico di SMPL) solo per leggerne le facce (la topologia dei triangoli):
        # le facce sono fisse per ogni modello SMPL, non dipendono dalla forma/posa specifica del corpo.
        self.faces = smplx.SMPL(self.body_model_path, ext="pkl").faces
        # Percorso al file JSON che associa ogni faccia della mesh a una parte del corpo (braccio, gamba, torace, ecc.),
        # necessario per filtrare i segmenti di taglio quando si misurano le circonferenze (vedi filter_body_part_slices).
        face_segmentation_path = os.path.join("data",
                                              f"{self.model_type}_body_parts_2_faces.json")
        # Carica effettivamente questa segmentazione da file.
        self.face_segmentation = load_face_segmentation(face_segmentation_path)

        # Dizionario {nome_landmark: indice_vertice} specifico per la topologia SMPL (6890 vertici).
        self.landmarks = SMPL_LANDMARK_INDICES
        # Dizionario che indica, per ogni misura, se e' di tipo LENGTH o CIRCUMFERENCE (da measurement_definitions.py).
        self.measurement_types = MEASUREMENT_TYPES
        # Definizioni delle misure di tipo lunghezza (coppie di landmark) specifiche per SMPL.
        self.length_definitions = SMPLMeasurementDefinitions().LENGTHS
        # Definizioni delle misure di tipo circonferenza (landmark + giunti per il piano di taglio) specifiche per SMPL.
        self.circumf_definitions = SMPLMeasurementDefinitions().CIRCUMFERENCES
        # Mappa che indica, per ogni circonferenza, quale parte del corpo usare per filtrare i segmenti di taglio.
        self.circumf_2_bodypart = SMPLMeasurementDefinitions().CIRCUMFERENCE_TO_BODYPARTS
        # Elenco di tutti i nomi di misura disponibili per il modello SMPL.
        self.all_possible_measurements = SMPLMeasurementDefinitions().possible_measurements

        # Mappa {nome_giunto: indice} per convertire i nomi dei giunti SMPL nei corrispondenti indici numerici.
        self.joint2ind = SMPL_JOINT2IND
        # Numero totale di giunti del modello SMPL (serve per costruire correttamente il body model).
        self.num_joints = SMPL_NUM_JOINTS

        # Numero di vertici della mesh SMPL: usato per validare le dimensioni di input in from_verts().
        self.num_points = 6890

    def from_verts(self,
                   verts: torch.tensor):
        '''
        Construct body model from only vertices.
        :param verts: torch.tensor (6890,3) of SMPL vertices
        '''        

        # Rimuove eventuali dimensioni "1" superflue (es. da (1,6890,3) a (6890,3)).
        verts = verts.squeeze()
        # Verifica che i vertici forniti abbiano davvero la forma attesa (num_points, 3);
        # in caso contrario interrompe l'esecuzione con un messaggio di errore chiaro (assert).
        error_msg = f"verts need to be of dimension ({self.num_points},3)"
        assert verts.shape == torch.Size([self.num_points,3]), error_msg

        # I giunti non sono salvati esplicitamente nei vertici: si ricavano da essi tramite un "regressore",
        # cioe' una matrice che, moltiplicata per i vertici, stima la posizione di ciascun giunto.
        joint_regressor = get_joint_regressor(self.model_type, 
                                              self.body_model_root,
                                              gender="NEUTRAL", 
                                              num_thetas=self.num_joints)
        # Applica il regressore ai vertici (moltiplicazione matriciale) per ottenere le posizioni dei giunti.
        joints = torch.matmul(joint_regressor, verts)
        # Converte i tensori PyTorch in array numpy (formato usato dal resto della classe per i calcoli geometrici) e li salva.
        self.joints = joints.numpy()
        self.verts = verts.numpy()

    def from_body_model(self,
                        gender: str,
                        shape: torch.tensor):
        '''
        Construct body model from given gender and shape params 
        of SMPl model.
        :param gender: str, MALE or FEMALE or NEUTRAL
        :param shape: torch.tensor, (1,10) beta parameters
                                    for SMPL model
        '''  

        # Crea (istanzia) il modello di corpo SMPL vero e proprio per il genere richiesto.
        model = create_model(model_type=self.model_type, 
                             model_root=self.body_model_root, 
                             gender=gender,
                             num_betas=10,
                             num_thetas=self.num_joints)    
        # Applica i parametri di forma (beta) al modello per ottenere vertici e giunti di un corpo con quella forma specifica.
        model_output = set_shape(model, shape)
        
        # Estrae vertici e giunti dal risultato, staccandoli dal grafo di calcolo di PyTorch (detach), portandoli su CPU
        # e convertendoli in array numpy; squeeze() rimuove le dimensioni di batch superflue.
        self.verts = model_output.vertices.detach().cpu().numpy().squeeze()
        self.joints = model_output.joints.squeeze().detach().cpu().numpy()
        self.gender = gender


class MeasureSMPLX(Measurer):
    '''
    Measure the SMPLX model defined either by the shape parameters or
    by its 10475 vertices. 

    All the measurements are expressed in cm.
    '''

    # Costruttore: identico nella struttura a MeasureSMPL.__init__, ma con i valori e i file specifici di SMPLX
    # (10475 vertici invece di 6890, file .npz invece di .pkl).
    def __init__(self):
        
        # Richiama il costruttore della classe base Measurer per inizializzare gli attributi comuni.
        super().__init__()

        # Nome del tipo di modello.
        self.model_type = "smplx"
        # Cartella radice dove sono salvati i file dei modelli di corpo.
        self.body_model_root = "data/body_models"
        # Costruisce il percorso completo alla sottocartella specifica del modello SMPLX (es. data/body_models/smplx).
        self.body_model_path = os.path.join(self.body_model_root, 
                                            self.model_type)

        # Carica il modello SMPLX solo per leggerne le facce. DIFFERENZA CHIAVE rispetto a SMPL:
        # qui si usa ext="npz" (formato numpy compresso), mentre SMPL usa ext="pkl" (formato pickle Python).
        self.faces = smplx.SMPLX(self.body_model_path, ext="npz").faces
        # Percorso al file JSON di segmentazione delle facce per parte del corpo, specifico per SMPLX.
        face_segmentation_path = os.path.join("data",
                                              f"{self.model_type}_body_parts_2_faces.json")
        # Carica la segmentazione da file.
        self.face_segmentation = load_face_segmentation(face_segmentation_path)

        # Dizionario {nome_landmark: indice_vertice} specifico per la topologia SMPLX (10475 vertici).
        self.landmarks = SMPLX_LANDMARK_INDICES
        # Dizionario tipo di misura (LENGTH/CIRCUMFERENCE), condiviso con SMPL.
        self.measurement_types = MEASUREMENT_TYPES
        # Definizioni delle misure di lunghezza specifiche per SMPLX.
        self.length_definitions = SMPLXMeasurementDefinitions().LENGTHS
        # Definizioni delle misure di circonferenza specifiche per SMPLX.
        self.circumf_definitions = SMPLXMeasurementDefinitions().CIRCUMFERENCES
        # Mappa circonferenza -> parte del corpo per SMPLX.
        self.circumf_2_bodypart = SMPLXMeasurementDefinitions().CIRCUMFERENCE_TO_BODYPARTS
        # Elenco di tutti i nomi di misura disponibili per SMPLX.
        self.all_possible_measurements = SMPLXMeasurementDefinitions().possible_measurements

        # Mappa {nome_giunto: indice} specifica per SMPLX.
        self.joint2ind = SMPLX_JOINT2IND
        # Numero totale di giunti del modello SMPLX.
        self.num_joints = SMPLX_NUM_JOINTS

        # Numero di vertici della mesh SMPLX (10475): usato per validare l'input in from_verts().
        self.num_points = 10475

    def from_verts(self,
                   verts: torch.tensor):
        '''
        Construct body model from only vertices.
        :param verts: torch.tensor (10475,3) of SMPLX vertices
        '''        

        # Rimuove eventuali dimensioni "1" superflue.
        verts = verts.squeeze()
        # Verifica che i vertici abbiano la forma attesa (num_points, 3), altrimenti solleva un errore.
        error_msg = f"verts need to be of dimension ({self.num_points},3)"
        assert verts.shape == torch.Size([self.num_points,3]), error_msg

        # Ricava i giunti dai vertici tramite il regressore di giunti specifico per SMPLX.
        joint_regressor = get_joint_regressor(self.model_type, 
                                              self.body_model_root,
                                              gender="NEUTRAL", 
                                              num_thetas=self.num_joints)
        # Applica il regressore ai vertici per ottenere le posizioni dei giunti.
        joints = torch.matmul(joint_regressor, verts)
        # Converte i risultati in array numpy e li salva sull'oggetto.
        self.joints = joints.numpy()
        self.verts = verts.numpy()

    def from_body_model(self,
                        gender: str,
                        shape: torch.tensor):
        '''
        Construct body model from given gender and shape params 
        of SMPl model.
        :param gender: str, MALE or FEMALE or NEUTRAL
        :param shape: torch.tensor, (1,10) beta parameters
                                    for SMPL model
        '''  

        # Crea il modello di corpo SMPLX per il genere richiesto.
        model = create_model(model_type=self.model_type, 
                             model_root=self.body_model_root, 
                             gender=gender,
                             num_betas=10,
                             num_thetas=self.num_joints)    
        # Applica i parametri di forma (beta) al modello per ottenere vertici e giunti con quella forma.
        model_output = set_shape(model, shape)
        
        # Estrae vertici e giunti, li stacca dal grafo di calcolo PyTorch, li porta su CPU e li converte in numpy.
        self.verts = model_output.vertices.detach().cpu().numpy().squeeze()
        self.joints = model_output.joints.squeeze().detach().cpu().numpy()
        self.gender = gender


class MeasureBody():
    # Pattern "factory": MeasureBody non e' pensata per essere usata come una classe normale.
    # __new__ viene eseguito PRIMA di __init__ e, invece di creare e restituire un'istanza di MeasureBody,
    # restituisce direttamente un'istanza di un'altra classe (MeasureSMPL o MeasureSMPLX).
    # Cosi' scrivendo `measurer = MeasureBody("smpl")` si ottiene gia' l'oggetto giusto,
    # scelto automaticamente in base alla stringa model_type passata.
    def __new__(cls, model_type):
        # Normalizza il testo in minuscolo, cosi' "SMPL", "Smpl", "smpl" vengono tutti riconosciuti.
        model_type = model_type.lower()
        # Se e' richiesto il modello SMPL, crea e restituisce un'istanza di MeasureSMPL.
        if model_type == 'smpl':
            return MeasureSMPL()
        # Se e' richiesto il modello SMPLX, crea e restituisce un'istanza di MeasureSMPLX.
        elif model_type == 'smplx':
            return MeasureSMPLX()
        # Qualsiasi altro valore di model_type non e' supportato: solleva un errore esplicito.
        else:
            raise NotImplementedError("Model type not defined")



# Questo blocco viene eseguito solo se il file e' lanciato direttamente (es. `python measure.py ...`),
# non quando measure.py viene importato come modulo da un altro script.
if __name__ == "__main__":

    # Configura un parser di argomenti da riga di comando, per scegliere quale modello misurare a titolo di test/demo.
    parser = argparse.ArgumentParser(description='Measure body models.')
    # Flag: se presente, misura un modello SMPL con forma "media" (tutti i parametri beta a zero).
    parser.add_argument('--measure_neutral_smpl_with_mean_shape', action='store_true',
                        help="Measure a mean shape smpl model.")
    # Flag: se presente, misura invece un modello SMPLX con forma "media".
    parser.add_argument('--measure_neutral_smplx_with_mean_shape', action='store_true',
                        help="Measure a mean shape smplx model.")
    # Legge effettivamente gli argomenti passati da riga di comando.
    args = parser.parse_args()

    # Lista dei tipi di modello da misurare in questa esecuzione (puo' contenere "smpl" e/o "smplx").
    model_types_to_measure = []
    # Aggiunge "smpl" e/o "smplx" alla lista a seconda dei flag passati da riga di comando.
    if args.measure_neutral_smpl_with_mean_shape:
        model_types_to_measure.append("smpl")
    elif args.measure_neutral_smplx_with_mean_shape:
        model_types_to_measure.append("smplx")

    # Per ogni tipo di modello richiesto, esegue l'intero flusso: crea il modello, lo misura, lo etichetta e lo visualizza.
    for model_type in model_types_to_measure:
        print(f"Measuring {model_type} body model")
        # Usa la factory MeasureBody per ottenere l'oggetto misuratore giusto (MeasureSMPL o MeasureSMPLX).
        measurer = MeasureBody(model_type)

        # Crea un vettore di 10 parametri di forma (beta) tutti a zero: corrisponde alla forma "media" del corpo.
        betas = torch.zeros((1, 10), dtype=torch.float32)
        # Costruisce il modello 3D (vertici e giunti) per un corpo di genere neutro con la forma media appena definita.
        measurer.from_body_model(gender="NEUTRAL", shape=betas)

        # Prende l'elenco di tutte le misure disponibili per questo tipo di modello.
        measurement_names = measurer.all_possible_measurements
        # Calcola effettivamente tutte le misure richieste.
        measurer.measure(measurement_names)
        # Stampa a schermo (in modo leggibile) il dizionario delle misure calcolate.
        print("Measurements")
        pprint(measurer.measurements)

        # Applica etichette standard (definite altrove come STANDARD_LABELS) alle misure, per un formato di output piu' leggibile.
        measurer.label_measurements(STANDARD_LABELS)
        # Stampa a schermo le misure con le etichette standard.
        print("Labeled measurements")
        pprint(measurer.labeled_measurements)

        # Apre la finestra di visualizzazione 3D interattiva con la mesh, i landmark, i giunti e le misure sovrapposti.
        measurer.visualize()