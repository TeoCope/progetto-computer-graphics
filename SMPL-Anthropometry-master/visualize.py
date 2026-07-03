
# --- Import delle librerie usate in questo file ---

# numpy: libreria per lavorare con array/matrici numerici (es. coordinate 3D dei vertici)
import numpy as np
# List: serve solo per i "type hint" (indicare che un parametro è una lista di stringhe, ecc.)
from typing import List
# plotly: libreria per creare grafici interattivi (qui usata per il tipo "Scatter3d" nelle annotazioni)
import plotly
# go ("graph_objects"): modulo di plotly con gli oggetti base per costruire i grafici (Mesh3d, Scatter3d, Figure...)
import plotly.graph_objects as go
# px ("express"): modulo di plotly usato qui solo per le palette di colori predefinite
import plotly.express as px
# trimesh: libreria per gestire mesh 3D (vertici + triangoli), usata per calcolare le sezioni del corpo
import trimesh
# argparse: libreria standard di Python per leggere argomenti passati da riga di comando
import argparse
# smplx: libreria che carica i modelli di corpo umano SMPL / SMPLX
import smplx
# json: libreria standard per leggere file in formato JSON (es. segmentazioni del corpo)
import json
# torch: libreria di PyTorch, usata per creare i tensori di input dei modelli (es. i parametri "betas")
import torch
#import ipywidgets as widgets
# make_subplots: funzione di plotly per creare una figura con più grafici affiancati (es. SMPL e SMPLX)
from plotly.subplots import make_subplots



# Import di altri moduli del progetto (file .py nella stessa cartella)
# MeasurementType: classe con le costanti che indicano il tipo di misura (LENGTH = lunghezza, CIRCUMFERENCE = circonferenza)
from measurement_definitions import MeasurementType
# Funzioni di utilità: calcolo dell'inviluppo convesso ("convex hull") di punti 3D
# e filtro delle sezioni della mesh appartenenti a una specifica parte del corpo
from anthro_utils import convex_hull_from_3D_points, filter_body_part_slices
# Dizionari {indice_giunto: nome_giunto} per i modelli SMPL e SMPLX
from joint_definitions import SMPL_IND2JOINT, SMPLX_IND2JOINT
# Dizionari {nome_landmark: indice_vertice} per i modelli SMPL e SMPLX
from landmark_definitions import SMPL_LANDMARK_INDICES, SMPLX_LANDMARK_INDICES

# Classe principale del file: si occupa di costruire il grafico 3D interattivo
# che mostra il corpo, i suoi giunti, i landmark anatomici e le misure calcolate.
class Visualizer():
    '''
    Visualize the body model with measurements, landmarks and joints.
    All the measurements are expressed in cm.
    '''

    # Costruttore della classe: riceve tutti i dati del corpo (vertici, facce, giunti,
    # landmark, misure...) e alcune opzioni booleane per decidere cosa disegnare.
    # I parametri sono molti ma servono solo a essere "salvati" come attributi dell'oggetto
    # (self.xxx = xxx) per poterli riusare più avanti nei metodi della classe.
    def __init__(self,
                 verts: np.ndarray,
                 faces: np.ndarray,
                 joints: np.ndarray,
                 landmarks: dict,
                 measurements: dict,
                 measurement_types: dict,
                 length_definitions: dict,
                 circumf_definitions: dict,
                 joint2ind: dict,
                 circumf_2_bodypart: dict,
                 face_segmentation: dict,
                 visualize_body: bool = True,
                 visualize_landmarks: bool = True,
                 visualize_joints: bool = True,
                 visualize_measurements: bool=True,
                 title: str = "Measurement visualization"
                ):


        # Dati geometrici e di misura del corpo: vengono semplicemente copiati
        # dai parametri di ingresso agli attributi dell'oggetto (self).
        self.verts = verts  # vertici della mesh: np.array (N,3) con le coordinate x,y,z
        self.faces = faces  # facce della mesh: np.array (F,3) con gli indici dei 3 vertici di ogni triangolo
        self.joints = joints  # coordinate 3D dei giunti (articolazioni) del corpo
        self.landmarks = landmarks  # dizionario {nome_landmark: indice_vertice_o_coppia_di_indici}
        self.measurements = measurements  # dizionario {nome_misura: valore_gia_calcolato_in_cm}
        self.measurement_types = measurement_types  # dizionario {nome_misura: tipo (LENGTH/CIRCUMFERENCE)}
        self.length_definitions = length_definitions  # dizionario con i landmark che delimitano ogni lunghezza
        self.circumf_definitions = circumf_definitions  # dizionario con landmark/giunti usati per ogni circonferenza
        self.joint2ind = joint2ind  # dizionario {nome_giunto: indice} per trovare l'indice numerico di un giunto
        self.circumf_2_bodypart = circumf_2_bodypart  # associa ogni misura di circonferenza alla parte del corpo
        self.face_segmentation = face_segmentation  # dizionario che associa le parti del corpo alle facce della mesh

        # Flag booleani: decidono quali elementi disegnare nel grafico finale.
        self.visualize_body = visualize_body  # True -> disegna la mesh (superficie) del corpo
        self.visualize_landmarks = visualize_landmarks  # True -> disegna i punti di riferimento anatomici
        self.visualize_joints = visualize_joints  # True -> disegna i giunti (articolazioni)
        self.visualize_measurements = visualize_measurements  # True -> disegna le misure (lunghezze/circonferenze)

        self.title = title  # titolo che verrà mostrato sul grafico



    # @staticmethod: metodo che non usa "self", cioè non dipende dallo stato
    # dell'oggetto Visualizer; si può chiamare anche senza creare un'istanza.
    @staticmethod
    def create_mesh_plot(verts: np.ndarray, faces: np.ndarray):
        '''
        Visualize smpl body mesh.
        :param verts: np.array (N,3) of vertices
        :param faces: np.array (F,3) of faces connecting the vertices

        Return:
        plotly Mesh3d object for plotting
        '''
        # go.Mesh3d crea una superficie 3D (la "pelle" del corpo) a partire da:
        # - le coordinate x,y,z dei vertici
        # - gli indici i,j,k che indicano quali 3 vertici formano ogni triangolo
        mesh_plot = go.Mesh3d(
                            x=verts[:,0],
                            y=verts[:,1],
                            z=verts[:,2],
                            color="gray",
                            # testo mostrato quando si passa il mouse sopra un punto: l'indice del vertice
                            hovertemplate ='<i>Index</i>: %{text}',
                            text = [i for i in range(verts.shape[0])],
                            # i, j and k give the vertices of triangles
                            i=faces[:,0],
                            j=faces[:,1],
                            k=faces[:,2],
                            opacity=0.6,  # trasparenza della mesh (0=invisibile, 1=opaca)
                            name='body',
                            )
        return mesh_plot

    # Metodo statico che disegna i giunti (le articolazioni) come punti neri a forma di croce.
    @staticmethod
    def create_joint_plot(joints: np.ndarray):

        # go.Scatter3d con mode='markers' disegna solo dei punti (marcatori), non linee,
        # nelle posizioni 3D dei giunti passati come argomento.
        return go.Scatter3d(x = joints[:,0],
                            y = joints[:,1],
                            z = joints[:,2],
                            mode='markers',
                            marker=dict(size=8,
                                        color="black",
                                        opacity=1,
                                        symbol="cross"
                                        ),
                            name="joints"
                                )

    # Metodo statico che costruisce il "wireframe" (reticolo di linee) della mesh,
    # cioè i bordi dei triangoli, utile per vedere la struttura della superficie.
    @staticmethod
    def create_wireframe_plot(verts: np.ndarray,faces: np.ndarray):
        '''
        Given vertices and faces, creates a wireframe of plotly segments.
        Used for visualizing the wireframe.
        
        :param verts: np.array (N,3) of vertices
        :param faces: np.array (F,3) of faces connecting the verts
        '''
        # Indici dei 3 vertici di ogni triangolo (i,j,k), presi dalle 3 colonne di "faces"
        i=faces[:,0]
        j=faces[:,1]
        k=faces[:,2]

        # Combina i tre gruppi di indici in un'unica matrice (F,3): una riga per triangolo
        triangles = np.vstack((i,j,k)).T

        # Coordinate x, y, z di tutti i vertici
        x=verts[:,0]
        y=verts[:,1]
        z=verts[:,2]

        # Ricompone x,y,z in una matrice (N,3) di punti 3D
        vertices = np.vstack((x,y,z)).T
        # Per ogni triangolo, recupera le coordinate dei suoi 3 vertici: shape (F,3,3)
        tri_points = vertices[triangles]

        #extract the lists of x, y, z coordinates of the triangle
        # vertices and connect them by a "line" by adding None
        # this is a plotly convention for plotting segments
        Xe = []
        Ye = []
        Ze = []
        # Per ogni triangolo si costruisce una "linea" che passa per i suoi 3 vertici
        # e torna al primo (range(4), con l'operatore % 3 che fa "girare" gli indici 0,1,2,0).
        # None separa un triangolo dal successivo, così plotly non disegna linee tra triangoli diversi.
        for T in tri_points:
            Xe.extend([T[k%3][0] for k in range(4)]+[ None])
            Ye.extend([T[k%3][1] for k in range(4)]+[ None])
            Ze.extend([T[k%3][2] for k in range(4)]+[ None])

        # return Xe, Ye, Ze
        # go.Scatter3d con mode='lines' disegna solo i segmenti (le linee), non i marcatori
        wireframe = go.Scatter3d(
                        x=Xe,
                        y=Ye,
                        z=Ze,
                        mode='lines',
                        name='wireframe',
                        line=dict(color= 'rgb(70,70,70)', width=1)
                        )
        return wireframe

    # Metodo (non statico, usa self.landmarks) che crea un punto 3D per ogni
    # landmark richiesto in landmark_names, restituendo una lista di oggetti plotly.
    def create_landmarks_plot(self,
                              landmark_names: List[str],
                              verts: np.ndarray
                              ) -> List[plotly.graph_objs.Scatter3d]:
        '''
        Visualize landmarks from landmark_names list
        :param landmark_names: List[str] of landmark names to visualize

        Return
        :param plots: list of plotly objects to plot
        '''

        plots = []  # lista che conterrà un oggetto grafico per ogni landmark

        # Associa a ogni nome di landmark un colore diverso, preso dalla palette "Alphabet"
        landmark_colors = dict(zip(self.landmarks.keys(),
                                px.colors.qualitative.Alphabet))

        # Ciclo su tutti i nomi di landmark richiesti dall'utente
        for lm_name in landmark_names:
            if lm_name not in self.landmarks.keys():
                print(f"Landmark {lm_name} is not defined.")
                pass

            # Un landmark può essere definito da un singolo indice di vertice,
            # oppure da una coppia (tuple) di indici: in tal caso si prende il punto medio.
            lm_index = self.landmarks[lm_name]
            if isinstance(lm_index,tuple):
                lm = (verts[lm_index[0]] + verts[lm_index[1]]) / 2
            else:
                lm = verts[lm_index]

            # Crea un singolo punto 3D (marker) nella posizione del landmark
            plot = go.Scatter3d(x = [lm[0]],
                                y = [lm[1]],
                                z = [lm[2]],
                                mode='markers',
                                marker=dict(size=8,
                                            color=landmark_colors[lm_name],
                                            opacity=1,
                                            ),
                               name=lm_name
                                )

            plots.append(plot)

        return plots

    # Metodo che disegna una misura di tipo "lunghezza" (LENGTH), cioè un segmento
    # che collega i due landmark che delimitano quella misura (es. altezza, lunghezza braccio).
    def create_measurement_length_plot(self,
                                       measurement_name: str,
                                       verts: np.ndarray,
                                       color: str
                                       ):
        '''
        Create length measurement plot.
        :param measurement_name: str, measurement name to plot
        :param verts: np.array (N,3) of vertices
        :param color: str of color to color the measurement

        Return
        plotly object to plot
        '''

        # Recupera i due landmark (indici di vertici) che definiscono questa misura di lunghezza
        measurement_landmarks_inds = self.length_definitions[measurement_name]

        # segments conterrà le coordinate x,y,z dei due estremi del segmento da disegnare
        segments = {"x":[],"y":[],"z":[]}
        for i in range(2):
            # Come sopra: se l'indice è una tupla si usa il punto medio tra i due vertici
            if isinstance(measurement_landmarks_inds[i],tuple):
                lm_tnp = (verts[measurement_landmarks_inds[i][0]] +
                          verts[measurement_landmarks_inds[i][1]]) / 2
            else:
                lm_tnp = verts[measurement_landmarks_inds[i]]
            segments["x"].append(lm_tnp[0])
            segments["y"].append(lm_tnp[1])
            segments["z"].append(lm_tnp[2])
        # Aggiunge un valore None in fondo a ogni lista di coordinate: è la convenzione
        # plotly per "chiudere" un segmento di linea (qui non strettamente necessaria
        # con un solo segmento, ma resta coerente con lo stile usato altrove nel file).
        for ax in ["x","y","z"]:
            segments[ax].append(None)

        # Se la misura è già stata calcolata, mostra il suo valore in cm nel nome
        # (visibile nella legenda del grafico); altrimenti mostra solo il nome della misura.
        if measurement_name in self.measurements:
            m_viz_name = f"{measurement_name}: {self.measurements[measurement_name]:.2f}cm"
        else:
            m_viz_name = measurement_name

        # Disegna il segmento come una linea 3D spessa (width=10) del colore indicato
        return go.Scatter3d(x=segments["x"],
                                    y=segments["y"],
                                    z=segments["z"],
                                    marker=dict(
                                            size=4,
                                            color="rgba(0,0,0,0)",
                                        ),
                                        line=dict(
                                            color=color,
                                            width=10),
                                        name=m_viz_name
                                        )

    # Metodo che disegna una misura di tipo "circonferenza" (CIRCUMFERENCE), cioè un
    # anello che gira intorno a una parte del corpo (es. giro vita, giro petto).
    # L'idea è: tagliare la mesh con un piano perpendicolare all'arto/tronco, ottenere
    # i segmenti dell'intersezione, e collegarli con un inviluppo convesso per formare
    # un anello chiuso.
    def create_measurement_circumference_plot(self,
                                              measurement_name: str,
                                              verts: np.ndarray,
                                              faces: np.ndarray,
                                              color: str):
        '''
        Create circumference measurement plot
        :param measurement_name: str, measurement name to plot
        :param verts: np.array (N,3) of vertices
        :param faces: np.array (F,3) of faces connecting the vertices
        :param color: str of color to color the measurement

        Return
        plotly object to plot
        '''

        # Recupera i landmark e i due giunti che definiscono il piano di taglio per questa circonferenza
        circumf_landmarks = self.circumf_definitions[measurement_name]["LANDMARKS"]
        circumf_landmark_indices = [self.landmarks[l_name] for l_name in circumf_landmarks]
        circumf_n1, circumf_n2 = self.circumf_definitions[measurement_name]["JOINTS"]
        circumf_n1, circumf_n2 = self.joint2ind[circumf_n1], self.joint2ind[circumf_n2]

        # Il piano di taglio passa per il punto medio dei landmark (plane_origin) ed è
        # orientato secondo la direzione tra i due giunti (plane_normal, il suo "vettore normale")
        plane_origin = np.mean(verts[circumf_landmark_indices,:],axis=0)
        plane_normal = self.joints[circumf_n1,:] - self.joints[circumf_n2,:]

        # Costruisce l'oggetto mesh di trimesh a partire da vertici e facce
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)

        # Calcola l'intersezione tra la mesh e il piano: restituisce i segmenti di taglio
        # (slice_segments) e gli indici delle facce coinvolte (sliced_faces)
        slice_segments, sliced_faces = trimesh.intersections.mesh_plane(mesh,
                        plane_normal=plane_normal,
                        plane_origin=plane_origin,
                        return_faces=True) # (N, 2, 3), (N,)

        # Tiene solo i segmenti di taglio che appartengono alla parte del corpo giusta
        # (es. per il giro vita, scarta eventuali intersezioni con le braccia)
        slice_segments = filter_body_part_slices(slice_segments,
                                                 sliced_faces,
                                                 measurement_name,
                                                 self.circumf_2_bodypart,
                                                 self.face_segmentation)

        # Ordina/collega i segmenti in un anello chiuso tramite l'inviluppo convesso
        slice_segments_hull = convex_hull_from_3D_points(slice_segments)


        # Prepara le liste di coordinate x,y,z da passare a plotly per disegnare
        # tutti i segmenti dell'anello, uno dopo l'altro (None separa un segmento dal successivo)
        draw_segments = {"x":[],"y":[],"z":[]}
        map_ax = {0:"x",1:"y",2:"z"}

        for i in range(slice_segments_hull.shape[0]):
            for j in range(3):
                draw_segments[map_ax[j]].append(slice_segments_hull[i,0,j])
                draw_segments[map_ax[j]].append(slice_segments_hull[i,1,j])
                draw_segments[map_ax[j]].append(None)

        # Come nel metodo precedente: mostra il valore numerico della misura, se disponibile
        if measurement_name in self.measurements:
            m_viz_name = f"{measurement_name}: {self.measurements[measurement_name]:.2f}cm"
        else:
            m_viz_name = measurement_name

        # Disegna l'anello come una sequenza di linee 3D (mode="lines") del colore indicato
        return go.Scatter3d(
                            x=draw_segments["x"],
                            y=draw_segments["y"],
                            z=draw_segments["z"],
                            mode="lines",
                            line=dict(
                                color=color,
                                width=10),
                            name=m_viz_name
                                )

    # Metodo principale della classe: mette insieme tutti i "pezzi" (mesh, wireframe,
    # giunti, landmark, misure) in un'unica figura plotly interattiva e la mostra a schermo.
    def visualize(self,
                  measurement_names: List[str] = [],
                  landmark_names: List[str] = [],
                  title="Measurement visualization"
                  ):
        '''
        Visualize the body model with measurements, landmarks and joints.

        :param measurement_names: List[str], list of strings with measurement names
        :param landmark_names: List[str], list of strings with landmark names
        :param title: str, title of plot
        '''

        # Crea una figura 3D vuota a cui aggiungere via via i vari elementi grafici (le "tracce")
        fig = go.Figure()

        if self.visualize_body:
            # visualize model mesh
            mesh_plot = self.create_mesh_plot(self.verts, self.faces)
            fig.add_trace(mesh_plot)
            # visualize wireframe
            wireframe_plot = self.create_wireframe_plot(self.verts, self.faces)
            fig.add_trace(wireframe_plot)

        # visualize joints
        if self.visualize_joints:
            joint_plot = self.create_joint_plot(self.joints)
            fig.add_trace(joint_plot)


        # visualize landmarks
        if self.visualize_landmarks:
            landmarks_plot = self.create_landmarks_plot(landmark_names, self.verts)
            fig.add_traces(landmarks_plot)


        # visualize measurements
        # Assegna un colore diverso a ogni tipo di misura, così ognuna è facilmente distinguibile
        measurement_colors = dict(zip(self.measurement_types.keys(),
                                  px.colors.qualitative.Alphabet))

        if self.visualize_measurements:
            # Per ogni misura richiesta, sceglie il metodo di disegno giusto
            # a seconda che sia una lunghezza o una circonferenza.
            for m_name in measurement_names:
                if m_name not in self.measurement_types.keys():
                    print(f"Measurement {m_name} not defined.")
                    pass

                if self.measurement_types[m_name] == MeasurementType().LENGTH:
                    measurement_plot = self.create_measurement_length_plot(measurement_name=m_name,
                                                                        verts=self.verts,
                                                                        color=measurement_colors[m_name])
                elif self.measurement_types[m_name] == MeasurementType().CIRCUMFERENCE:
                    measurement_plot = self.create_measurement_circumference_plot(measurement_name=m_name,
                                                                                    verts=self.verts,
                                                                                    faces=self.faces,
                                                                                    color=measurement_colors[m_name])

                fig.add_trace(measurement_plot)


        # Imposta le dimensioni della figura, il titolo e "scene_aspectmode='data'"
        # (mantiene le proporzioni reali degli assi x,y,z, senza deformare il corpo)
        fig.update_layout(scene_aspectmode='data',
                            width=1000, height=700,
                            title=title,
                            )

        # Mostra la figura interattiva (si apre di solito nel browser o in una finestra)
        fig.show()


# Funzione "standalone" (non fa parte della classe Visualizer) che crea un modello
# SMPLX di prova in posa neutra e ne disegna i giunti, uno per uno, con un nome ed
# un colore diverso, opzionalmente insieme alla mesh del corpo.
def viz_smplx_joints(visualize_body=True,fig=None,show=True,title="SMPLX joints"):
    """
    Visualize smpl joints on the same plot.
    :param visualize_body: bool, whether to visualize the body or not.
    :param fig: plotly Figure object, if None, create new figure.
    """

    # "betas" sono i parametri di forma del corpo: qui tutti a zero -> corpo "medio"/standard
    betas = torch.zeros((1, 10), dtype=torch.float32)

    # Carica il modello SMPLX (neutro, senza dettagli del viso) dalla cartella "data"
    smplx_model =  smplx.create(model_path="data",
                                model_type="smplx",
                                gender="NEUTRAL",
                                use_face_contour=False,
                                num_betas=10,
                                #body_pose=torch.zeros((1, (55-1) * 3)),
                                ext='pkl')

    # Esegue il modello con i betas scelti, ottenendo giunti e vertici del corpo risultante
    smplx_model = smplx_model(betas=betas, return_verts=True)
    # .detach().numpy() converte il tensore PyTorch in un array numpy "staccato" dal calcolo dei gradienti
    smplx_joints = smplx_model.joints.detach().numpy()[0]
    # Il giunto 0 è il bacino (pelvis): si sottrae la sua posizione a tutti i giunti
    # per centrare il corpo nell'origine (0,0,0) del grafico
    smplx_joint_pelvis = smplx_joints[0,:]
    smplx_joints = smplx_joints - smplx_joint_pelvis
    smplx_vertices = smplx_model.vertices.detach().numpy()[0]
    smplx_vertices = smplx_vertices - smplx_joint_pelvis
    # Le facce (la topologia della mesh) non dipendono dalla posa: si caricano a parte
    smplx_faces = smplx.SMPLX("data/smplx",ext="pkl").faces

    # Costruisce una lunga lista di colori concatenando più palette, per avere
    # abbastanza colori diversi da assegnare a ciascun giunto (l'ultimo è nero)
    joint_colors = px.colors.qualitative.Alphabet + \
                   px.colors.qualitative.Dark24 + \
                   px.colors.qualitative.Alphabet + \
                   px.colors.qualitative.Dark24 + \
                   px.colors.qualitative.Alphabet + \
                   ["#000000"]

    # Se non viene passata una figura esistente, ne crea una nuova
    if isinstance(fig,type(None)):
        fig = go.Figure()

    # Disegna un punto per ogni giunto del modello SMPLX
    for i in range(smplx_joints.shape[0]):

        # Se il giunto ha un nome noto (definito in joint_definitions.py) lo usa,
        # altrimenti gli assegna un nome generico "noname-<indice>"
        if i in SMPLX_IND2JOINT.keys():
            joint_name = SMPLX_IND2JOINT[i]
        else:
            joint_name = f"noname-{i}"

        joint_plot = go.Scatter3d(x = [smplx_joints[i,0]],
                                    y = [smplx_joints[i,1]],
                                    z = [smplx_joints[i,2]],
                                    mode='markers',
                                    marker=dict(size=10,
                                                color=joint_colors[i],
                                                opacity=1,
                                                symbol="circle"
                                                ),
                                    name="smplx-"+joint_name
                                        )


        fig.add_trace(joint_plot)


    # Se richiesto, aggiunge anche la mesh del corpo (in rosso, semi-trasparente)
    if visualize_body:
        plot_body = go.Mesh3d(
                            x=smplx_vertices[:,0],
                            y=smplx_vertices[:,1],
                            z=smplx_vertices[:,2],
                            color = "red",
                            i=smplx_faces[:,0],
                            j=smplx_faces[:,1],
                            k=smplx_faces[:,2],
                            name='smplx mesh',
                            showscale=True,
                            opacity=0.5
                        )
        fig.add_trace(plot_body)

    fig.update_layout(scene_aspectmode='data',
                        width=1000, height=700,
                        title=title,
                        )


    # Se show=True mostra subito il grafico, altrimenti restituisce la figura
    # (utile per aggiungere altre tracce prima di mostrarla, es. in viz_smpl_joints)
    if show:
        fig.show()
    else:
        return fig


# Equivalente della funzione precedente ma per il modello SMPL (invece di SMPLX):
# stessa logica, corpo in blu, simbolo "cross" per i giunti invece di "circle".
def viz_smpl_joints(visualize_body=True,fig=None,show=True,title="SMPL joints"):
    """
    Visualize smpl joints on the same plot.
    :param visualize_body: bool, whether to visualize the body or not.
    :param fig: plotly Figure object, if None, create new figure.
    """

    betas = torch.zeros((1, 10), dtype=torch.float32)

    smpl_model =  smplx.create(model_path="data",
                                model_type="smpl",
                                gender="NEUTRAL",
                                use_face_contour=False,
                                num_betas=10,
                                ext='pkl')

    smpl_model = smpl_model(betas=betas, return_verts=True)
    smpl_joints = smpl_model.joints.detach().numpy()[0]
    smpl_joints_pelvis = smpl_joints[0,:]
    # Centra il corpo sottraendo la posizione del bacino, come fatto per SMPLX
    smpl_joints = smpl_joints - smpl_joints_pelvis
    smpl_vertices = smpl_model.vertices.detach().numpy()[0]
    smpl_vertices = smpl_vertices - smpl_joints_pelvis
    smpl_faces = smplx.SMPL("data/smpl",ext="pkl").faces


    joint_colors = px.colors.qualitative.Alphabet + \
                   px.colors.qualitative.Dark24 + \
                   px.colors.qualitative.Alphabet + \
                   px.colors.qualitative.Dark24 + \
                   px.colors.qualitative.Alphabet + \
                   ["#000000"]

    if isinstance(fig,type(None)):
        fig = go.Figure()

    for i in range(smpl_joints.shape[0]):

        if i in SMPL_IND2JOINT.keys():
            joint_name = SMPL_IND2JOINT[i]
        else:
            joint_name = f"noname-{i}"

        joint_plot = go.Scatter3d(x = [smpl_joints[i,0]],
                                    y = [smpl_joints[i,1]],
                                    z = [smpl_joints[i,2]],
                                    mode='markers',
                                    marker=dict(size=10,
                                                color=joint_colors[i],
                                                opacity=1,
                                                symbol="cross"
                                                ),
                                    name="smpl-"+joint_name
                                        )


        fig.add_trace(joint_plot)

    if visualize_body:
        plot_body = go.Mesh3d(
                            x=smpl_vertices[:,0],
                            y=smpl_vertices[:,1],
                            z=smpl_vertices[:,2],
                            #facecolor=face_colors,
                            color = "blue",
                            i=smpl_faces[:,0],
                            j=smpl_faces[:,1],
                            k=smpl_faces[:,2],
                            name='smpl mesh',
                            showscale=True,
                            opacity=0.5
                        )
        fig.add_trace(plot_body)

    fig.update_layout(scene_aspectmode='data',
                        width=1000, height=700,
                        title=title,
                        )
    if show:
        fig.show()
    else:
        return fig



# Funzione generica che disegna una mesh colorando ogni triangolo (faccia) secondo
# la lista di colori "face_colors": serve per visualizzare la segmentazione del corpo
# in parti (es. braccio, gamba, testa...), ognuna con un colore diverso.
def viz_face_segmentation(verts,faces,face_colors,
                          title="Segmented body",name="mesh",show=True):
    """
    Visualize face segmentation defined in face_colors.
    :param verts: np.ndarray - (N,3) representing the vertices
    :param faces: np.ndarray - (F,3) representing the indices of the faces
    :param face_colors: np.ndarray - (F,3) representing the colors of the faces
    """

    import plotly.graph_objects as go

    fig = go.Figure()
    # A differenza di create_mesh_plot, qui si usa "facecolor" (colore per ogni
    # singola faccia) invece di "color" (colore unico per tutta la mesh)
    mesh_plot = go.Mesh3d(
            x=verts[:,0],
            y=verts[:,1],
            z=verts[:,2],
            facecolor=face_colors,
            i=faces[:,0],
            j=faces[:,1],
            k=faces[:,2],
            name=name,
            showscale=True,
            opacity=1
        )
    fig.add_trace(mesh_plot)
    fig.update_layout(scene_aspectmode='data',
                        width=1000, height=700,
                        title=title)

    if show:
        fig.show()
    else:
        return fig


# Carica il modello SMPL, la segmentazione delle facce per parte del corpo (salvata
# in un file JSON) e prepara i colori da passare a viz_face_segmentation.
def viz_smpl_face_segmentation(fig=None, show=True, title="SMPL face segmentation"):
    body = smplx.SMPL("data/smpl",ext="pkl")

    # face_segmentation è un dizionario {nome_parte_del_corpo: lista_indici_facce}
    with open("data/smpl/smpl_body_parts_2_faces.json","r") as f:
        face_segmentation = json.load(f)

    faces = body.faces
    verts = body.v_template  # "v_template" = vertici del corpo in posa neutra (senza deformazioni)

    # create colors for each face
    colors = px.colors.qualitative.Alphabet + \
            px.colors.qualitative.Dark24
    # Associa a ogni nome di parte del corpo un numero progressivo (0,1,2,...)
    mapping_bp2ind = dict(zip(face_segmentation.keys(),
                            range(len(face_segmentation.keys()))
                            ))
    # Inizializza la lista dei colori delle facce (una entry per ogni faccia della mesh)
    face_colors = [0]*faces.shape[0]
    # Per ogni parte del corpo, colora tutte le sue facce con lo stesso colore
    for bp_name,bp_indices in face_segmentation.items():
        bp_label = mapping_bp2ind[bp_name]
        for i in bp_indices:
            face_colors[i] = colors[bp_label]

    if isinstance(fig,type(None)):
        fig = go.Figure()

    fig = viz_face_segmentation(verts,faces,face_colors,title=title,name="smpl",show=False)

    if show:
        fig.show()
    else:
        return fig


# Stessa logica della funzione precedente, ma per il modello SMPLX
# (file JSON di segmentazione diverso, cartella dati diversa).
def viz_smplx_face_segmentation(fig=None,show=True,title="SMPLX face segmentation"):
    """
    Visualize face segmentations for smplx.
    """

    body = smplx.SMPLX("data/smplx",ext="pkl")

    with open("data/smplx/smplx_body_parts_2_faces.json","r") as f:
        face_segmentation = json.load(f)


    faces = body.faces
    verts = body.v_template

    # create colors for each face
    colors = px.colors.qualitative.Alphabet + \
            px.colors.qualitative.Dark24
    mapping_bp2ind = dict(zip(face_segmentation.keys(),
                            range(len(face_segmentation.keys()))
                            ))
    face_colors = [0]*faces.shape[0]
    for bp_name,bp_indices in face_segmentation.items():
        bp_label = mapping_bp2ind[bp_name]
        for i in bp_indices:
            face_colors[i] = colors[bp_label]

    if isinstance(fig,type(None)):
        fig = go.Figure()

    fig = viz_face_segmentation(verts,faces,face_colors,title=title,name="smpl",show=False)

    if show:
        fig.show()
    else:
        return fig


# Funzione generica che disegna i vertici raggruppati per parte del corpo
# (a differenza di viz_face_segmentation, qui si colorano dei punti, non i triangoli).
def viz_point_segmentation(verts,point_segm,title="Segmented body",fig=None,show=True):
    """
    Visualze points and their segmentation defined in dict point_segm.
    :param verts: np.ndarray - (N,3) representing the vertices
    :param point_segm: dict - dict mapping body part to all points belonging
                                to it
    """
    colors = px.colors.qualitative.Alphabet + \
             px.colors.qualitative.Dark24

    if isinstance(fig,type(None)):
        fig = go.Figure()

    # Per ogni parte del corpo disegna tutti i suoi vertici con lo stesso colore,
    # così ogni parte è riconoscibile a colpo d'occhio nel grafico
    for i, (body_part, body_indices) in enumerate(point_segm.items()):
        plot = go.Scatter3d(x = verts[body_indices,0],
                            y = verts[body_indices,1],
                            z = verts[body_indices,2],
                            mode='markers',
                            marker=dict(size=5,
                                        color=colors[i],
                                        opacity=1,
                                        #symbol="cross"
                                        ),
                            name=body_part
                                )
        fig.add_trace(plot)
    fig.update_layout(scene_aspectmode='data',
                    width=1000, height=700,
                    title=title)
    if show:
        fig.show()
    return fig


# Carica il modello SMPLX e la sua segmentazione "per punti" (file JSON), poi
# richiama viz_point_segmentation per disegnarla.
def viz_smplx_point_segmentation(fig=None,show=True,title="SMPLX point segmentation"):
    """
    Visualize point segmentations for smplx.
    """

    model_path = "data/smplx"
    smpl_verts = smplx.SMPLX(model_path,ext="pkl").v_template
    with open("data/smplx/point_segmentation_meshcapade.json","r") as f:
        point_segm = json.load(f)
    fig = viz_point_segmentation(smpl_verts,point_segm,title=title,fig=fig,show=show)

    if show:
        fig.show()
    else:
        return fig


# Equivalente della funzione precedente, ma per il modello SMPL.
def viz_smpl_point_segmentation(fig=None,show=True,title="SMPL point segmentation"):
    """
    Visualize point segmentations for smpl.
    """

    model_path = "data/smpl"
    smpl_verts = smplx.SMPL(model_path,ext="pkl").v_template
    with open("data/smpl/point_segmentation_meshcapade.json","r") as f:
        point_segm = json.load(f)
    fig = viz_point_segmentation(smpl_verts,point_segm,title=title,fig=fig,show=show)

    if show:
        fig.show()
    else:
        return fig


# Disegna tutti i vertici della mesh come piccoli punti grigi e semitrasparenti
# (utile per esplorare/individuare gli indici dei vertici), e in più evidenzia
# con punti grandi e colorati i landmark passati in landmark_dict.
def viz_landmarks(verts,landmark_dict,title="Visualize landmarks",fig=None,show=True,name="points"):

    if isinstance(fig,type(None)):
        fig = go.Figure()

    # Nuvola di tutti i vertici, semitrasparente: passando il mouse sopra un punto
    # se ne vede l'indice (hovertemplate), utile per scoprire l'indice di un vertice
    plot = go.Scatter3d(x = verts[:,0],
                        y = verts[:,1],
                        z = verts[:,2],
                        mode='markers',
                        hovertemplate ='<i>Index</i>: %{text}',
                        text = [i for i in range(verts.shape[0])],
                        marker=dict(size=5,
                                    color="black",
                                    opacity=0.2,
                                    # line=dict(color='black',width=1)
                                    ),
                        name=name
                            )

    fig.add_trace(plot)

    colors = px.colors.qualitative.Alphabet + \
             px.colors.qualitative.Dark24  + \
             px.colors.qualitative.Alphabet + \
             px.colors.qualitative.Dark24

    # Per ogni landmark definito, disegna un punto grande a forma di croce,
    # con un colore diverso e un nome nella legenda (prefisso "name-")
    for i, (lm_name, lm_ind) in enumerate(landmark_dict.items()):
        plot = go.Scatter3d(x = [verts[lm_ind,0]],
                            y = [verts[lm_ind,1]],
                            z = [verts[lm_ind,2]],
                            mode='markers',
                            marker=dict(size=10,
                                        color=colors[i],
                                        opacity=1,
                                        symbol="cross"
                                        ),
                            name=name+"-"+lm_name
                                )
        fig.add_trace(plot)

    fig.update_layout(scene_aspectmode='data',
                    width=1000, height=700,
                    title=title)

    if show:
        fig.show()
    else:
        return fig


# Carica i vertici "template" del modello SMPL e i suoi landmark predefiniti,
# poi li disegna con viz_landmarks.
def viz_smpl_landmarks(fig=None,show=True,title="SMPL landmarks"):
    """
    Visualize smpl landmarks.
    """

    verts = smplx.SMPL("data/smpl",ext="pkl").v_template
    landmark_dict = SMPL_LANDMARK_INDICES

    if isinstance(fig,type(None)):
        fig=go.Figure()

    fig = viz_landmarks(verts,
                        landmark_dict,
                        title="Visualize landmarks",
                        fig=fig,
                        show=show,
                        name="smpl")

    if show:
        fig.show()
    else:
        return fig


# Equivalente della funzione precedente, ma per il modello SMPLX.
def viz_smplx_landmarks(fig=None,show=True,title="SMPLX landmarks"):
    """
    Visualize smplx landmarks.
    """

    verts = smplx.SMPLX("data/smplx",ext="pkl").v_template
    landmark_dict = SMPLX_LANDMARK_INDICES

    if isinstance(fig,type(None)):
        fig=go.Figure()

    fig = viz_landmarks(verts,
                        landmark_dict,
                        title="Visualize landmarks",
                        fig=fig,
                        show=show,
                        name="smplx")

    if show:
        fig.show()
    else:
        return fig



# Questo blocco viene eseguito solo se il file viene lanciato direttamente
# (es. "python visualize.py --opzione"), non quando viene importato da un altro file.
# Serve come piccola "demo a riga di comando" per provare le funzioni sopra definite.
if __name__ == "__main__":

    # Definisce gli argomenti accettati da riga di comando: sono tutti flag booleani
    # (action='store_true' -> se presenti valgono True, altrimenti False di default)
    parser = argparse.ArgumentParser(description='Visualize body models, joints and segmentations..')
    parser.add_argument('--visualize_smpl_and_smplx_face_segmentation', action='store_true',
                        help="Visualize face segmentations for smplx model.")
    parser.add_argument('--visualize_smpl_and_smplx_joints', action='store_true',
                        help="visualize smpl and smplx joints on same plot.")
    parser.add_argument('--visualize_smpl_and_smplx_point_segmentation', action='store_true',
                        help="visualize smpl and smplx point segmentation on two separate plots.")
    parser.add_argument('--visualize_smpl_and_smplx_landmarks', action='store_true',
                        help="visualize smpl and smplx landmarks on two separate plots.")
    args = parser.parse_args()  # legge gli argomenti effettivamente passati da terminale



    if args.visualize_smpl_and_smplx_face_segmentation:
        # mesh is not compatible with subplots so these are plotted
        # onto separate plots
        viz_smpl_face_segmentation(fig=None, show=True)
        viz_smplx_face_segmentation(fig=None,show=True)



    if args.visualize_smpl_and_smplx_joints:
        # Disegna prima i giunti/mesh di SMPL (show=False, non mostra ancora il grafico),
        # poi aggiunge quelli di SMPLX sulla STESSA figura e infine la mostra (show=True)
        title = "SMPL and SMPLX joints"
        fig = viz_smpl_joints(visualize_body=True,
                              fig=None,
                              show=False,
                              title=title)
        viz_smplx_joints(visualize_body=True,
                        fig=fig,
                        show=True,
                        title=title)

    if args.visualize_smpl_and_smplx_point_segmentation:
        # Qui invece SMPL e SMPLX vengono mostrati in due sotto-grafici (subplot)
        # affiancati, ciascuno con la propria scena 3D ('type': 'scene')
        fig = make_subplots(rows=1, cols=2,
                            specs=[[{'type': 'scene'},
                                    {'type': 'scene'}]],
                            subplot_titles=("SMPL", "SMPLX"))
        title="SMPL and SMPLX point segmentation"


        # Genera le due figure separatamente (show=False: non le mostra ancora)
        fig_smpl = viz_smpl_point_segmentation(fig=None,show=False,title=title)
        fig_smplx = viz_smplx_point_segmentation(fig=None,show=False,title=title)


        # Copia tutte le "tracce" (i singoli elementi grafici) di ciascuna figura
        # nella colonna corrispondente della figura combinata con i subplot
        for i in range(len(fig_smpl.data)):
            fig.add_trace(fig_smpl.data[i],row=1,col=1)
        for i in range(len(fig_smplx.data)):
            fig.add_trace(fig_smplx.data[i],row=1,col=2)


        fig.update_layout(fig_smpl.layout)
        fig.update_layout(scene2_aspectmode="data",
                          showlegend=False,
                          width=1200,
                          height=700)
        fig.show()

    if args.visualize_smpl_and_smplx_landmarks:
        # Stessa logica del blocco precedente, ma per i landmark invece che per la segmentazione
        fig = make_subplots(rows=1, cols=2,
                            specs=[[{'type': 'scene'},
                                    {'type': 'scene'}]],
                            subplot_titles=("SMPL", "SMPLX"))
        title="SMPL and SMPLX landmarks"

        fig_smpl = viz_smpl_landmarks(fig=None,show=False,title=title)
        fig_smplx = viz_smplx_landmarks(fig=None,show=False,title=title)

        for i in range(len(fig_smpl.data)):
            fig.add_trace(fig_smpl.data[i],row=1,col=1)
        for i in range(len(fig_smplx.data)):
            fig.add_trace(fig_smplx.data[i],row=1,col=2)


        fig.update_layout(fig_smpl.layout)
        fig.update_layout(scene2_aspectmode="data",
                          showlegend=False,
                          width=1200,
                          height=700)
        fig.show()