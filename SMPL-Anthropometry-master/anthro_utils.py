
# NOTA: questo file si chiamava originariamente `utils.py`, ma è stato rinominato
# in `anthro_utils.py` per evitare conflitti di import con il file `utils.py`
# del progetto principale (altrimenti Python importerebbe il file sbagliato).

# Modulo json: serve per leggere/scrivere file .json (es. le segmentazioni della mesh)
import json
# Modulo sys: usato qui solo per uscire dal programma con un messaggio di errore
import sys
# numpy: libreria per lavorare con array/matrici in modo efficiente
import numpy as np
# ConvexHull: funzione di scipy che calcola l'inviluppo convesso (convex hull)
# di un insieme di punti, cioè il più piccolo poligono convesso che li contiene tutti
from scipy.spatial import ConvexHull
# os: modulo per operazioni sul sistema operativo (qui non viene usato direttamente,
# ma è importato comunque)
import os
# argparse: libreria standard per gestire gli argomenti da riga di comando (CLI)
import argparse

def load_face_segmentation(path: str):
        '''
        Carica la segmentazione delle facce che definisce, per ogni parte
        del modello corporeo, le facce che le appartengono.
        :param path: str - percorso del file json con la segmentazione delle facce
        '''

        # proviamo ad aprire il file indicato da "path" in modalità lettura ('r')
        try:
            with open(path, 'r') as f:
                # carichiamo il contenuto JSON e lo trasformiamo in un dizionario Python
                face_segmentation = json.load(f)
        except FileNotFoundError:
            # se il file non esiste, interrompiamo il programma stampando un messaggio chiaro
            sys.exit(f"No such file - {path}")

        # restituiamo il dizionario {parte del corpo: lista di facce} appena caricato
        return face_segmentation


def convex_hull_from_3D_points(slice_segments: np.ndarray):
        '''
        Crea l'inviluppo convesso (convex hull) a partire da punti 3D
        :param slice_segments: np.ndarray, dim N x 2 x 3 che rappresenta N segmenti 3D

        Restituisce:
        :param slice_segments_hull: np.ndarray, dim N x 2 x 3 che rappresenta gli N segmenti 3D
                                    che formano l'inviluppo convesso
        '''

        # "slice_segments" è un array di N segmenti, ogni segmento fatto da 2 punti 3D
        # (è il risultato del taglio della mesh con un piano, es. per misurare una circonferenza).
        # stack all points in N x 3 array
        # np.concatenate unisce tutti i segmenti in un'unica lista di punti 3D
        # (da N x 2 x 3 diventa (N*2) x 3)
        merged_segment_points = np.concatenate(slice_segments)
        # eliminiamo eventuali punti duplicati (due segmenti adiacenti condividono un punto)
        unique_segment_points = np.unique(merged_segment_points,
                                            axis=0)

        # points lie in plane -- find which ax of x,y,z is redundant
        # Dato che i punti del taglio giacciono tutti su uno stesso piano (es. un piano
        # orizzontale per la circonferenza vita), una delle 3 coordinate (x, y o z) varia
        # pochissimo tra tutti i punti: è la coordinata "ridondante" (quella perpendicolare
        # al piano di taglio). Per trovarla calcoliamo, per ogni asse, la differenza tra il
        # valore massimo e il valore minimo dei punti; l'asse con la differenza più piccola
        # (idealmente vicina a zero) è l'asse "appiattito" da scartare.
        # np.argmin restituisce l'indice (0=x, 1=y, 2=z) dell'asse con la variazione minima.
        redundant_plane_coord = np.argmin(np.max(unique_segment_points,axis=0) -
                                            np.min(unique_segment_points,axis=0) )
        # teniamo solo gli altri due assi (quelli su cui i punti si "sparpagliano"),
        # perché sono quelli su cui possiamo calcolare un poligono 2D
        non_redundant_coords = [x for x in range(3) if x!=redundant_plane_coord]

        # create convex hull
        # calcoliamo il convex hull 2D usando solo le due coordinate non ridondanti:
        # in questo modo lavoriamo su un problema 2D invece che 3D (più semplice e robusto)
        hull = ConvexHull(unique_segment_points[:,non_redundant_coords])
        # hull.simplices contiene, per ogni lato del poligono convesso, gli indici dei due
        # punti che lo formano (nel sistema di riferimento 2D); li appiattiamo in un
        # vettore 1D di indici
        segment_point_hull_inds = hull.simplices.reshape(-1)

        # usiamo quegli indici per recuperare i punti 3D originali (non più solo le 2
        # coordinate proiettate), così da avere di nuovo dei segmenti in 3D
        slice_segments_hull = unique_segment_points[segment_point_hull_inds]
        # ricostruiamo la forma N x 2 x 3 (N segmenti, 2 punti ciascuno, 3 coordinate)
        slice_segments_hull = slice_segments_hull.reshape(-1,2,3)

        # restituiamo i segmenti 3D che formano il perimetro (convex hull) della sezione
        return slice_segments_hull


def filter_body_part_slices(slice_segments:np.ndarray,
                             sliced_faces:np.ndarray,
                             measurement_name: str,
                             circumf_2_bodypart: dict,
                             face_segmentation: dict
                            ):
        '''
        Rimuove i segmenti che non appartengono alla parte del corpo
        corretta per la misura data.
        :param slice_segments: np.ndarray - (N,2,3) per N segmenti
                                            rappresentati come due punti 3D
        :param sliced_faces: np.ndarray - (N,) che rappresenta gli indici delle
                                            facce
        :param measurement_name: str - nome della misura
        :param circumf_2_bodypart: dict - dizionario che mappa la misura alla parte del corpo
        :param face_segmentation: dict - dizionario che mappa la parte del corpo a tutte le
                                        facce che le appartengono

        Restituisce:
        :param slice_segments: np.ndarray (K,2,3) con K < N, per K segmenti
                                rappresentati come due punti 3D che si trovano nella
                                parte del corpo corretta
        '''

        # controlliamo se la misura richiesta (es. "waist circumference") ha una parte
        # del corpo associata nel dizionario circumf_2_bodypart
        if measurement_name in circumf_2_bodypart.keys():

            # recuperiamo la/le parte/i del corpo corrispondenti a questa misura
            # (es. "waist" -> "spine1", oppure una lista di parti)
            body_parts = circumf_2_bodypart[measurement_name]

            # se body_parts è una lista di più parti del corpo...
            if isinstance(body_parts,list):
                # ...uniamo in un'unica lista tutte le facce che appartengono a
                # ciascuna di quelle parti del corpo (list comprehension annidata)
                body_part_faces = [face_index for body_part in body_parts
                                    for face_index in face_segmentation[body_part]]
            else:
                # altrimenti body_parts è una singola parte del corpo: prendiamo
                # direttamente le sue facce dal dizionario di segmentazione
                body_part_faces = face_segmentation[body_parts]

            # numero totale di segmenti/facce ottenuti dal taglio della mesh
            N_sliced_faces = sliced_faces.shape[0]

            # qui accumuliamo gli indici dei segmenti da mantenere
            keep_segments = []
            # scorriamo ogni faccia tagliata...
            for i in range(N_sliced_faces):
                # ...e teniamo il segmento solo se la sua faccia appartiene alla parte
                # del corpo corretta (es. escludiamo segmenti delle braccia dal calcolo
                # della circonferenza vita, anche se numericamente intersecano lo stesso piano)
                if sliced_faces[i] in body_part_faces:
                    keep_segments.append(i)

            # restituiamo solo i segmenti filtrati (indicizzando con la lista di indici)
            return slice_segments[keep_segments]

        else:
            # se la misura non ha una parte del corpo specifica associata, non filtriamo
            # nulla e restituiamo tutti i segmenti così come sono
            return slice_segments


def point_segmentation_to_face_segmentation(
                point_segmentation: dict,
                faces: np.ndarray,
                save_as: str = None):
    """
    :param point_segmentation: dict - dizionario che mappa la parte del corpo a
                                      tutti i punti che le appartengono
    :param faces: np.ndarray - (N,3) che rappresenta gli indici delle facce
    :param save_as: str - percorso opzionale dove salvare la segmentazione delle
                          facce come json
    """
    # Questa funzione è un'utility "offline": si usa una tantum per generare i file
    # JSON di segmentazione per faccia a partire da una segmentazione per vertice
    # (cioè: sappiamo a quale parte del corpo appartiene ogni singolo vertice/punto
    # della mesh, e vogliamo dedurre a quale parte appartiene ogni faccia/triangolo).

    # import locali, usati solo dentro questa funzione
    import json
    # tqdm mostra una barra di avanzamento nel ciclo (utile perché può essere lento)
    from tqdm import tqdm
    # Counter serve per contare le occorrenze di elementi in una lista
    from collections import Counter

    # create body parts to index mapping
    # creiamo una mappa che associa ad ogni nome di parte del corpo un numero intero
    # (indice), così possiamo lavorare con numeri invece che stringhe (più efficiente)
    mapping_bp2ind = dict(zip(point_segmentation.keys(),
                              range(len(point_segmentation.keys()))))
    # creiamo anche la mappa inversa: da indice numerico a nome della parte del corpo
    mapping_ind2bp = {v:k for k,v in mapping_bp2ind.items()}


    # assign each face to body part index
    # creiamo un array della stessa forma di "faces" (N facce x 3 vertici), inizializzato
    # a zero, che conterrà per ogni vertice di ogni faccia l'indice della parte del corpo
    faces_segmentation = np.zeros_like(faces)
    # scorriamo ogni faccia (i = indice della faccia, face = i 3 indici di vertice)
    for i,face in tqdm(enumerate(faces)):
        # per ogni parte del corpo e il relativo elenco di indici di punti/vertici...
        for bp_name, bp_indices in point_segmentation.items():
            # ...recuperiamo l'etichetta numerica di questa parte del corpo
            bp_label = mapping_bp2ind[bp_name]

            # controlliamo ciascuno dei 3 vertici che compongono la faccia (un triangolo)
            for k in range(3):
                # se il k-esimo vertice della faccia appartiene a questa parte del corpo...
                if face[k] in bp_indices:
                    # ...assegniamo l'etichetta di quella parte del corpo a quel vertice
                    faces_segmentation[i,k] = bp_label


    # for each face, assign the most common body part
    # ogni faccia ha 3 vertici che potrebbero (raramente) appartenere a parti del corpo
    # diverse: per decidere a quale parte assegnare l'intera faccia, prendiamo la parte
    # del corpo più frequente (la "maggioranza") tra i suoi 3 vertici
    face_segmentation_final = np.zeros(faces_segmentation.shape[0])
    for i,f in enumerate(faces_segmentation):
        # contiamo quante volte compare ogni etichetta tra i 3 vertici della faccia
        c = Counter(list(f))
        # c.most_common()[0][0] è l'etichetta più frequente; la assegniamo alla faccia i
        face_segmentation_final[i] = c.most_common()[0][0]


    # create dict with body part as key and faces as values
    # costruiamo il dizionario finale: per ogni parte del corpo, una lista vuota che
    # riempiremo con gli indici delle facce che le appartengono
    face_segmentation_dict = {k:[] for k in mapping_bp2ind.keys()}
    for i,fff in enumerate(face_segmentation_final):
        # convertiamo l'etichetta numerica nel nome della parte del corpo (mapping_ind2bp)
        # e aggiungiamo l'indice della faccia i alla lista corrispondente
        face_segmentation_dict[mapping_ind2bp[int(fff)]].append(i)


    # save face segmentation
    # se è stato passato un percorso "save_as", salviamo il dizionario come file JSON
    if save_as:
        with open(save_as, 'w') as f:
            json.dump(face_segmentation_dict, f)

    # restituiamo comunque il dizionario, anche se non è stato salvato su disco
    return face_segmentation_dict


# questo blocco viene eseguito solo se il file viene lanciato direttamente
# (es. `python anthro_utils.py --create_face_segmentation`), non quando viene importato
if __name__ == "__main__":

    # creiamo un parser di argomenti da riga di comando con una breve descrizione
    parser = argparse.ArgumentParser(description='Create face segmentation from \
                                     point segmentation of smpl/smplx models.')
    # definiamo un'opzione booleana: se presente sulla riga di comando, il suo valore
    # sarà True, altrimenti False (action='store_true')
    parser.add_argument('--create_face_segmentation', action='store_true')
    # leggiamo gli argomenti effettivamente passati da riga di comando
    args = parser.parse_args()

    # se l'utente ha specificato --create_face_segmentation...
    if args.create_face_segmentation:

        # importiamo smplx solo qui, perché serve solo in questo ramo del codice
        import smplx

        # percorso del file JSON con la segmentazione per punto/vertice del modello SMPLX
        segm_path = "data/smplx/point_segmentation_meshcapade.json"
        with open(segm_path,"r") as f:
            # carichiamo la segmentazione per vertice come dizionario Python
            point_segmentation = json.load(f)

        # cartella dove si trovano i file del modello SMPLX
        model_path = "data/smplx"
        # carichiamo il modello SMPLX (formato .pkl) e prendiamo le sue facce (triangoli)
        smplx_faces = smplx.SMPLX(model_path,ext="pkl").faces

        # percorso dove salveremo il risultato: la segmentazione per faccia
        save_as = "data/smplx/smplx_body_parts_2_faces.json"

        # eseguiamo la conversione da segmentazione per vertice a segmentazione per faccia
        # e salviamo il risultato nel file indicato da save_as
        _ = point_segmentation_to_face_segmentation(point_segmentation,
                                                    smplx_faces,
                                                    save_as)
