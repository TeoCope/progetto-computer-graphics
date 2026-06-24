# Calcolo delle Conversioni e dei Parametri Corporei

Questo documento descrive in dettaglio come avvengono le conversioni tra i parametri di forma dell'avatar (coefficienti `betas`) e le misure antropometriche reali (es. altezza, circonferenze), identificando i file e le funzioni in cui tali calcoli sono implementati.

---

## 1. Conversione Diretta: da Betas a Misure Fisiche (in cm)

Questa conversione calcola le misure fisiche reali dell'avatar partendo dai parametri di forma `betas` ed è implementata nel modulo `SMPL-Anthropometry-master`.

### File Principale
- [measure.py](file:///Users/matteocopertari/Progetti/progetto-computer-graphics/SMPL-Anthropometry-master/measure.py)

### Funzionamento
Il processo si divide in base al tipo di misura (definito in [measurement_definitions.py](file:///Users/matteocopertari/Progetti/progetto-computer-graphics/SMPL-Anthropometry-master/measurement_definitions.py)):

#### A. Misura di Lunghezza (es. altezza, lunghezza braccio)
- **Funzione**: `measure_length(self, measurement_name: str)`
- **Calcolo**: 
  1. Identifica i landmark associati alla misura (ad esempio, per l'altezza, la sommità della testa e i talloni, definiti in [landmark_definitions.py](file:///Users/matteocopertari/Progetti/progetto-computer-graphics/SMPL-Anthropometry-master/landmark_definitions.py)).
  2. Recupera le coordinate 3D dei vertici corrispondenti sulla mesh dell'avatar.
  3. Calcola la distanza euclidea $d$ tra i punti in metri e la converte in centimetri:
     $$d = \sqrt{(x_2-x_1)^2 + (y_2-y_1)^2 + (z_2-z_1)^2} \times 100$$

#### B. Misura di Circonferenza (es. vita, torace, fianchi)
- **Funzione**: `measure_circumference(self, measurement_name: str)`
- **Calcolo**:
  1. Identifica i landmark di origine del piano di taglio e i giunti (joints) che definiscono la normale al piano (es. per il torace, i capezzoli come origine e l'asse del busto come normale).
  2. Taglia la mesh dell'avatar con questo piano 3D usando la funzione `trimesh.intersections.mesh_plane`.
  3. Filtra le sezioni trovate tramite la funzione `filter_body_part_slices` (in [anthro_utils.py](file:///Users/matteocopertari/Progetti/progetto-computer-graphics/SMPL-Anthropometry-master/anthro_utils.py)), per tenere solo i segmenti del corpo desiderati (es. escludendo le braccia per la circonferenza torace).
  4. Crea un poligono convesso a partire dai segmenti estratti tramite `convex_hull_from_3D_points` (in [anthro_utils.py](file:///Users/matteocopertari/Progetti/progetto-computer-graphics/SMPL-Anthropometry-master/anthro_utils.py)).
  5. Calcola il perimetro totale del convex hull in centimetri, che rappresenta la circonferenza corporea.

---

## 2. Conversione Inversa: da Misure Fisiche Target a Betas (Ottimizzazione)

Questa conversione calcola i coefficienti di forma `betas` dell'avatar affinché la sua corporatura rispecchi le misure desiderate inserite dall'utente.

### File Principale
- [main.py](file:///Users/matteocopertari/Progetti/progetto-computer-graphics/main.py)

### Funzioni Coinvolte
- **`_on_fit_measurements(self)`**: Legge i valori inseriti dall'utente e pianifica l'esecuzione dell'ottimizzazione sul thread principale.
- **`_run_optimization_fit(self, body_model_name, gender_name, targets)`**: Esegue l'ottimizzazione numerica tramite SciPy.

### Dettagli del Calcolo
L'ottimizzatore utilizza l'algoritmo **L-BFGS-B** della funzione `scipy.optimize.minimize` per minimizzare una funzione di perdita (loss):

1. **Inizializzazione**: Parte dai cursori dei `betas` correnti per convergere più rapidamente.
2. **Definizione della Loss**: 
   La funzione `fit_loss(betas_np)` esegue i seguenti passaggi ad ogni iterazione:
   - Converte i `betas` correnti in un tensor PyTorch ed esegue il pass in avanti (forward pass) del modello SMPL/SMPLX per calcolare la nuova mesh (vertici e giunti).
   - Applica la traslazione per poggiare i piedi sul piano terra.
   - Chiama il misuratore per estrarre le misure reali corrispondenti ai target dell'utente.
   - Calcola l'errore quadratico totale:
     $$\text{Loss} = \sum_{i} (\text{MisuraCalcolata}_i - \text{MisuraTarget}_i)^2$$
3. **Calcolo dei Gradienti tramite Differenze Finite**:
   Poiché le operazioni geometriche (taglio di piani e convex hull) non sono analiticamente differenziabili, SciPy stima i gradienti perturbando i parametri. Abbiamo impostato la perturbazione `eps = 0.1` per superare i limiti di precisione geometrica locale e ottenere gradienti affidabili.
4. **Aggiornamento GUI**:
   Trovato il minimo della loss, la funzione imposta i cursori grafici sui nuovi `betas` calcolati e ricarica il modello.
