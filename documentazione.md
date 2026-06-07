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
- **Target Measurements (cm)**: Campi di input numerici in cui inserire le misure antropometriche desiderate (Altezza, Circonferenza Petto, Circonferenza Vita, Circonferenza Fianchi, Altezza Interno Gamba, Lunghezza Braccio).
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
