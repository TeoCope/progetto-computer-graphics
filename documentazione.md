# Documentazione e Guida per l'Utente – Simulatore di Avatar Umani Realistici

Questo documento descrive lo sviluppo, l'architettura e l'utilizzo del **Simulatore di Avatar Umani Realistici** basato sui modelli parametrici SMPL/SMPLX e integrato con un sistema di misurazione antropometrica e ottimizzazione di forma.

---

## 1. Architettura del Sistema e Sviluppo

Il sistema unisce due componenti principali:
1. **Visualizzatore 3D Interattivo (Open3D)**: Interfaccia grafica che permette di ruotare la telecamera, regolare i parametri del modello (betas, espressioni, pose), visualizzare gli scheletri dei joint ed eseguire risolutori IK (Inverse Kinematics).
2. **Misuratore Antropometrico (SMPL-Anthropometry)**: Motore di calcolo geometrico (basato su `trimesh` e `scipy`) che calcola in tempo reale le lunghezze (distanze euclidee tra landmark) e le circonferenze (sezioni piane tridimensionali e calcolo del convex hull) sul corpo dell'avatar.

### Integrazione e Ottimizzazioni Eseguite:
- **Risoluzione delle collisioni di importazione (Import Shadowing)**: Il modulo `SMPL-Anthropometry-master/utils.py` è stato rinominato in `anthro_utils.py` per evitare conflitti con il file `utils.py` del progetto principale.
- **Supporto dinamico alle estensioni dei modelli**: Risolto un bug nel caricamento di SMPLX in `measure.py` aggiungendo il supporto per i file `.npz` oltre ai tradizionali `.pkl` di SMPL.
- **Caching geometrico per i render di overlay**: Modificate le funzioni `measure_length` e `measure_circumference` in `measure.py` per salvare le coordinate 3D dei segmenti calcolati. In questo modo il visualizzatore Open3D può disegnare le sezioni piane e le linee direttamente sull'avatar 3D.
- **Risolutore di fitting con SciPy**: Implementato un ottimizzatore basato sull'algoritmo **L-BFGS-B** (da `scipy.optimize.minimize`). Per risolvere il problema della non-differenziabilità delle operazioni discrete (come il convex hull e le intersezioni mesh-piano), abbiamo configurato il passo delle differenze finite (`eps`) a `0.1`. Questo consente al solutore di ricevere gradienti numerici puliti ed eseguire il fitting completo in circa **1-2 secondi**.

---

## 2. Funzionalità Principali dell'Interfaccia

Il visualizzatore estende la barra laterale destra con la nuova sezione **"Measurements"**:
- **Target Measurements (cm)**: Campi di input numerici in cui inserire le misure antropometriche desiderate. Sono disponibili tutte le misure mostrate in *Current Measurements* (Altezza, Circonferenze di Petto/Vita/Fianchi/Collo/Testa, Altezza Interno Gamba, Lunghezza Braccio, Larghezza Spalle); i campi lasciati a 0 non vincolano il fitting.
- **Fit Model to Targets (Bottone)**: Avvia l'ottimizzazione automatica che calcola i 10 parametri di forma (`betas`) ideali per far coincidere le misure dell'avatar con i target inseriti.
- **Visible Overlay (Dropdown)**: Permette di selezionare una misura specifica e visualizzarla graficamente in tempo reale sul modello 3D (linee rosse per le circonferenze corporee, linee rosse con endpoint verdi per le altezze/lunghezze).
- **Current Measurements (Lista)**: Elenco aggiornato in tempo reale che mostra le misure effettive dell'avatar in centimetri (es. `Waist: 89.5 cm`) ad ogni modifica dei cursori o al termine del fitting.
- **Menu di Esportazione 3D**: Nel menu principale del programma (`File > Export 3D Mesh (OBJ)`), l'utente può esportare la mesh finale dell'avatar, comprensiva di personalizzazioni e pose, in formato Wavefront OBJ.

---

## 3. Guida all'Uso e Installazione

### Requisiti ed Installazione
Per eseguire il simulatore, assicurarsi di utilizzare l'ambiente conda preconfigurato `computer_graphics`. Se necessario, attivarlo e verificare le dipendenze:
```bash
conda activate computer_graphics
# Le dipendenze trimesh e plotly sono già state configurate ed installate nell'ambiente
```

### Avvio dell'Applicazione
Lanciare il visualizzatore eseguendo il seguente comando dalla root del progetto:
```bash
conda run -n computer_graphics python main.py
```
*Opzionale: aggiungere il flag `--web` se si desidera abilitare lo streaming WebRTC nel browser.*

### Passi per Personalizzare ed Esportare un Avatar
1. **Scegliere il Modello e Genere**: Nella sezione *Model settings*, selezionare il modello del corpo (SMPL o SMPLX) e il genere (Neutral, Male, Female).
2. **Visualizzare le Misure Attuali**: Espandere la sezione *Measurements* per vedere le misure del modello a riposo (T-pose).
3. **Controllare Shape, Posa ed Espressioni**:
   - Usare i cursori dei *Betas* per modificare manualmente caratteristiche come stazza, proporzioni e larghezza spalle. Le misure attuali si aggiorneranno in tempo reale.
   - Sotto *Pose Controls*, selezionare un joint ed impostare gli angoli di rotazione o cliccare su *Run IK* per allineare i giunti.
4. **Visualizzazione Overlay**:
   - Nel menu *Visible Overlay*, selezionare ad esempio `chest circumference` per visualizzare la sezione piana rossa all'altezza del petto.
5. **Eseguire il Fitting Automatico**:
   - Digitare i valori target desiderati nei campi (es. Altezza: `180`, Vita: `85`, Fianchi: `98`).
   - Cliccare su **Fit Model to Targets**. Lo schermo mostrerà il messaggio *"Running optimization... Please wait"* e dopo 1-2 secondi caricherà il corpo perfetto ottimizzando i cursori.
6. **Esportare il Modello**:
   - Cliccare su **File > Export 3D Mesh (OBJ)** per salvare l'avatar in formato OBJ per future elaborazioni o prove virtuali (virtual try-on) in programmi esterni (es. Blender, Unity).

---

## 4. Preset di Avatar, Texture e Integrazione Unity

### Preset antropometrici (sezione "Avatar Preset")
La barra laterale ora si apre con la sezione **Avatar Preset**, che offre quattro profili predefiniti basati su caratteristiche antropometriche e demografiche realistiche (misure coerenti con i rispettivi BMI):
- *Uomo adulto magro* (178 cm, BMI ~19)
- *Uomo adulto robusto* (176 cm, BMI ~29)
- *Donna giovane normopeso* (165 cm, BMI ~21)
- *Donna adulta robusta* (162 cm, BMI ~29)

Ogni preset è definito in `presets.json` da un insieme di **misure target** (altezza, circonferenze di torace/vita/fianchi, interno gamba); le betas corrispondenti sono precalcolate dallo script offline `build_presets.py` con lo stesso ottimizzatore L-BFGS-B del fitting interattivo (funzione condivisa nel modulo `fitting.py`), con errori residui tipicamente sotto 1-2 cm. Selezionando un profilo e premendo **Apply Preset** l'avatar assume genere, forma e misure del preset; i campi target della sezione *Measurements* vengono precompilati, così l'utente può partire dal profilo e personalizzarlo (alla prima modifica manuale il menu torna su "Custom"). **Reset Avatar** riporta corpo, posa ed espressione allo stato iniziale.

Per rigenerare le betas dopo aver modificato le misure in `presets.json`:
```bash
python build_presets.py --force
```

### Texture dell'avatar
Il menu **Texture** (sempre nella sezione Avatar Preset) applica al corpo SMPL una texture fotorealistica (pelle, volto, capelli). Richiede il template UV ufficiale `smpl_uv.obj` e immagini texture nel layout SMPL, da collocare in `data/textures/` (istruzioni di download in `data/textures/README.md`). Le coordinate UV sono per-triangolo, per gestire le cuciture del template; l'export OBJ include UV, file `.mtl` e texture.

### Integrazione Unity
- La cartella `unity/` contiene gli script C# (`AvatarController`, `UIManager`, `AvatarPresets`, `FitServiceClient`) che replicano l'interfaccia in Unity sopra il pacchetto ufficiale **SMPL-X for Unity** di MPI, con aggiornamento della forma **in tempo reale** tramite blendshapes; i preset sono letti dallo **stesso `presets.json`** dell'app Python. Poiché gli spazi di forma di SMPL e SMPL-X non sono compatibili, `build_presets.py` calcola per ogni preset anche la variante `betas_smplx` usata da Unity.
- Come nella GUI Python, **l'utente definisce l'avatar inserendo le misure in cm**: il bottone *Fit to Measurements* invia le misure al server locale `avatar_server.py` (root del repo, avvio con `python avatar_server.py`), che riusa **esattamente** il misuratore geometrico `MeasureBody` e l'ottimizzatore L-BFGS-B condivisi in `fitting.py` e risponde con le betas e l'errore residuo. Lo stesso server calcola le *Current Measurements* reali del corpo corrente mostrate in Unity (endpoint `POST /measure`, usato anche al movimento degli slider). Questa architettura garantisce che input antropometrico e avatar siano coerenti in entrambe le applicazioni, senza duplicare la logica in C#.
- **File > Save Model Params** salva, accanto al file joblib, anche un **JSON portabile** (`model_type`, `gender`, `betas`, misure correnti, eventuale `preset_id`), caricabile in Unity con "Load JSON". Setup completo in `unity/README.md`.

---

## 5. Note sul Modello Matematico delle Espressioni (FLAME / SMPL-X)

Il modello SMPL-X integra il modello facciale **FLAME**, che permette la deformazione del viso tramite blendshapes definiti statisticamente.

A differenza dei tradizionali sistemi di animazione facciale (come le *Action Units* del sistema FACS o gli *ARKit blendshapes* di Apple), che cercano di isolare il movimento di singoli muscoli (es. "solleva solo l'angolo destro del labbro"), i parametri di espressione di SMPL-X (identificati comunemente come *Exp 0, Exp 1...*) rappresentano le **Componenti Principali (PCA)** estratte dall'analisi di migliaia di scansioni 3D di volti umani reali.

Questo significa che l'algoritmo matematico ha catturato la **co-occorrenza naturale** dei movimenti facciali umani. Di conseguenza:
- Modificando un singolo parametro (es. lo slider che abbiamo rinominato in "Sorriso / Broncio"), non si muoveranno unicamente le labbra, ma verranno coinvolte anche le guance (che si sollevano) e gli occhi (che tendono a socchiudersi leggermente).
- Un parametro legato all'apertura della bocca (es. "Sorpresa") modificherà contemporaneamente la mandibola, l'apertura oculare e l'inclinazione delle sopracciglia.

Questo approccio olistico e statistico, pur apparendo meno "granulare" nel controllo del singolo dettaglio muscolare, garantisce che ogni combinazione di parametri generi sempre espressioni **anatomicamente plausibili e realistiche**, evitando distorsioni innaturali (i cosiddetti "artefatti") tipiche dei sistemi a blendshape isolati tradizionali. 

Le etichette descrittive assegnate agli slider nella UI indicano quindi semplicemente l'**effetto visivo dominante** associato a ciascuna Componente Principale, fungendo da guida per l'utente finale.
