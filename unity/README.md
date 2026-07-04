# Avatar SMPL in Unity

Questa cartella contiene un **progetto Unity completo** che riproduce
l'interfaccia dell'app Python: selezione dei preset di avatar, **definizione
dell'avatar tramite le misure in cm** (come nella GUI del main), aggiornamento
del modello SMPL-X **in tempo reale** tramite blendshapes, texture,
espressioni facciali e riproduzione di **animazioni mocap** (AMASS).

Il fitting misure→betas NON è riscritto in C#: Unity dialoga con un piccolo
server locale (`avatar_server.py`, nella root del repo) che riusa la stessa
identica logica della GUI Python — misuratore geometrico `MeasureBody` +
ottimizzatore L-BFGS-B del modulo `fitting.py`. Il server serve anche le
misure reali del corpo corrente per il pannello "Current Measurements".

## Cosa è già incluso nel progetto

Il progetto è versionato per intero (Assets, Packages, ProjectSettings), non
serve alcun import manuale:

- **SMPL-X for Unity** (pacchetto ufficiale MPI) in `Assets/SMPLX/`: prefab
  SMPL-X maschile/femminile/neutro con blendshapes, scheletro e lo script
  `SMPLX.cs` che i nostri script richiamano. Licenza: solo uso di
  ricerca/didattica (vedi `Assets/SMPLX/LICENSE.md`), da citare nella
  presentazione.
- **bmlSUP (SMPL Unity Player)** come pacchetto locale in
  `Packages/com.biomotionlab.sup/`: fornisce le animazioni mocap di esempio
  in formato JSON (convertite da AMASS) in
  `Packages/com.biomotionlab.sup/SampleAnimations/JsonAnimations/`.
- **Newtonsoft Json** dichiarato in `Packages/manifest.json` (serve per
  leggere i dizionari di misure di `presets.json` e del server).
- **`Assets/StreamingAssets/presets.json`**: copia del `presets.json` della
  root del repo. **Rifare la copia ogni volta che si rilancia
  `build_presets.py`**: contiene sia le betas SMPL sia quelle SMPL-X
  (`betas_smplx`, quelle usate qui).

## Requisiti

- **Unity 6** (il progetto è stato creato con `6000.5.2f1`, vedi
  `ProjectSettings/ProjectVersion.txt`): aprire direttamente la cartella
  `unity/` da Unity Hub.
- **Server di fitting** (per il fit dalle misure): dalla root del repo
  ```bash
  python avatar_server.py
  ```
  Lasciarlo in esecuzione mentre si usa Unity. Senza server restano comunque
  funzionanti preset, texture, espressioni e animazioni (usano dati locali);
  non funzionano il bottone "Adatta alle Misure" e l'aggiornamento delle
  Current Measurements.
- **Texture (opzionale)**: il dropdown Texture elenca le immagini
  (`.png`/`.jpg`) presenti in `data/textures/` nella root del repo — la
  stessa cartella usata dall'app Python (istruzioni per il download in
  `data/textures/README.md`; per Unity non serve `smpl_uv.obj`, le UV sono
  già nei prefab SMPL-X).

## Setup della scena (passi da fare nell'Editor)

1. Trascinare in scena i prefab SMPL-X **maschile** e **femminile**
   (es. `Assets/SMPLX/Models/smplx-male` e `smplx-female`), posizionarli
   nell'origine.
2. Creare un GameObject vuoto `AvatarManager` e aggiungergli i componenti
   `AvatarController`, `UIManager`, `FitServiceClient` e
   `SUPAnimationBridge`.
3. In `AvatarController` assegnare i riferimenti `maleAvatar`/`femaleAvatar`
   (il componente `SMPLX` dei due prefab in scena); in `SUPAnimationBridge`
   assegnare `avatarController`.
4. Costruire la Canvas uGUI con il menu Editor **`Avatar SMPL > Build
   Settings Canvas`** (script `Assets/Editor/BuildAvatarUI.cs`). Crea una
   `Canvas` con un pannello scorrevole a sinistra contenente tutti i widget
   (preset, genere, texture, misure, espressioni, animazioni, reset), già
   dimensionati e organizzati in righe etichettate. Se in scena esiste già
   il GameObject `AvatarManager` (punto 2), lo script collega **anche
   automaticamente** tutti i riferimenti dell'Inspector di `UIManager`,
   incluse le 9 chiavi di `Measurement Fields` e l'`animationBridge`;
   restano da assegnare a mano solo `avatarController` e `fitClient`
   (riferimenti ad altri componenti, non a widget creati dallo script).
   Rilanciabile: ricrea la Canvas da zero ogni volta.
   Nota: la canvas automatica non include più gli slider betas né il
   caricamento JSON; `UIManager` li supporta ancora se si creano i widget a
   mano e si assegnano `betaSliders`/`jsonPathInput`/`loadJsonButton`.
5. Avviare `python avatar_server.py` nella root del repo, poi Play.

## La UI a runtime

Tutte le etichette sono in italiano. Dall'alto verso il basso:

- **Preset**: dropdown con "Personalizzato" + i profili di `presets.json`;
  selezionandone uno il corpo cambia istantaneamente (betas `betas_smplx`
  precalcolate), genere e campi misura si allineano al profilo.
- **Genere** (Maschio/Femmina): attiva il prefab corrispondente e vi
  riapplica la texture corrente (betas ed espressioni sono proprie di
  ciascun prefab).
- **Texture**: dropdown ("Nessuna" + le immagini trovate in
  `data/textures/`); la texture è caricata dal disco a runtime e applicata
  al materiale della mesh, e viene riapplicata al cambio genere.
- **Misure desiderate (cm)**: i 9 campi (altezza, circonferenze di
  torace/vita/fianchi/collo/testa, interno gamba, lunghezza braccio,
  larghezza spalle) + bottone **"Adatta alle Misure"**: invia i target al
  server, che esegue il fitting reale (qualche decina di secondi) e
  risponde con le betas; il corpo si aggiorna e lo stato mostra l'errore
  residuo massimo. Il pannello **Current Measurements** è ricalcolato dal
  server (misuratore geometrico vero) a ogni modifica.
- **Espressioni Facciali**: 10 slider (range -2..+2) che pilotano i
  blendshape di espressione FLAME di SMPL-X in tempo reale, con etichette
  descrittive dell'effetto dominante di ogni componente.
- **Animazioni (SUP)**: dropdown con le animazioni di esempio del pacchetto
  bmlSUP (nomi italiani: Danza 1/2, Conversazione, Pugilato) e bottone
  **Riproduci/Ferma** (toggle, verde/rosso). La riproduzione avviene tramite
  `SUPAnimationBridge`, che legge il JSON AMASS (quaternioni per 52 joint
  SMPL-H + traslazioni del bacino) e lo ritargettizza sullo scheletro
  SMPL-X: mappa i nomi delle ossa, converte le coordinate dal sistema
  destrorso di SMPL a quello sinistrorso di Unity e al frame 0 corregge
  orientamento e offset così che l'avatar parta di fronte alla camera e
  alla giusta altezza. Velocità (`playbackSpeed`) e `loop` sono regolabili
  dall'Inspector; l'animazione segue il cambio genere e la corporatura
  corrente, e "Ferma" riporta l'avatar alla posa di riposo.
- **Reimposta**: azzera betas ed espressioni (corpo medio) e svuota i campi.

## Interscambio con l'app Python

- **Preset**: entrambe le app leggono lo stesso `presets.json` (qui la copia
  in `Assets/StreamingAssets/`). Le betas dei preset sono precalcolate da
  `build_presets.py` con il fitting sulle misure antropometriche target, in
  due varianti: `betas` (modello SMPL, usato dalla GUI Python) e
  `betas_smplx` (modello SMPL-X, usato qui). Servono due fit separati perché
  gli spazi di forma di SMPL e SMPL-X non sono compatibili.
- **Texture**: stessa cartella `data/textures/` per entrambe le app.
- **Fitting**: il bottone "Adatta alle Misure" usa il server locale, quindi
  misure e betas sono calcolate ESATTAMENTE come nella GUI Python (stesse
  funzioni di `fitting.py` e `measure.py`), sul modello SMPL-X del genere
  selezionato.
- **Avatar personalizzati**: nell'app Python `File > Save Model Params`
  salva, accanto al file joblib, un `.json` con genere e betas. Se nella
  scena si aggiungono i widget `jsonPathInput` + `loadJsonButton` (non
  inclusi nella canvas automatica), incollando il percorso e premendo "Load
  JSON" si ricrea l'avatar in Unity. Per una corrispondenza fedele impostare
  il Body Model su **SMPLX** nell'app Python prima del fit e dell'export
  (con betas SMPL la forma su SMPL-X è solo approssimata; la UI mostra un
  avviso).

## Come funziona il tempo reale

Il pacchetto MPI espone in `SMPLX.cs` gli array `betas[10]` ed
`expressions[10]`: ogni parametro pilota un blendshape della mesh e
`UpdateJointPositions()` ricalcola lo scheletro per la nuova corporatura.
I nostri script si limitano a scrivere i parametri e chiamare questi metodi,
quindi ogni aggiornamento (preset, fit dal server, slider) modifica la mesh
nello stesso frame; solo il calcolo del fit avviene fuori da Unity, nel
server Python. Le animazioni invece non passano dai blendshapes: il bridge
scrive direttamente le rotazioni locali delle ossa dello scheletro a ogni
`LateUpdate`, quindi convivono con qualunque corporatura/texture corrente.
