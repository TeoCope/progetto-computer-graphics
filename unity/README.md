# Avatar SMPL in Unity

Questa cartella contiene gli script C# per riprodurre in Unity l'interfaccia
dell'app Python: selezione dei preset di avatar, **definizione dell'avatar
tramite le misure in cm** (come nella GUI del main) e aggiornamento del
modello SMPL-X **in tempo reale** tramite blendshapes.

Il fitting misure→betas NON è riscritto in C#: Unity dialoga con un piccolo
server locale (`avatar_server.py`, nella root del repo) che riusa la stessa
identica logica della GUI Python — misuratore geometrico `MeasureBody` +
ottimizzatore L-BFGS-B del modulo `fitting.py`. Il server serve anche le
misure reali del corpo corrente per il pannello "Current Measurements".

## Requisiti

- **Unity 2021.3 LTS o successiva** (render pipeline Built-in).
- **SMPL-X for Unity** (pacchetto ufficiale MPI): scaricare lo
  `.unitypackage` da <https://smpl-x.is.tue.mpg.de> (sezione Downloads,
  serve l'account già usato per i modelli `.pkl`). Contiene i prefab
  SMPL-X maschile/femminile/neutro con blendshapes, scheletro, texture di
  esempio e lo script `SMPLX.cs` che i nostri script richiamano.
  Licenza: solo uso di ricerca/didattica, da citare nella presentazione.
- **Newtonsoft Json**: in Unity, `Window > Package Manager > + > Add package
  by name...` e inserire `com.unity.nuget.newtonsoft-json`
  (serve per leggere i dizionari di misure di `presets.json` e del server).
- **Server di fitting** (per il fit dalle misure): dalla root del repo
  ```bash
  python avatar_server.py
  ```
  Lasciarlo in esecuzione mentre si usa Unity. Senza server restano comunque
  funzionanti i preset e gli slider betas (usano le betas precalcolate);
  non funzionano il bottone "Fit to Measurements" e l'aggiornamento delle
  Current Measurements.

## Setup del progetto (passi da fare nell'Editor)

1. Creare un nuovo progetto Unity 3D (Built-in render pipeline) dentro
   questa cartella `unity/` (oppure crearlo altrove e copiarci `Assets/Scripts`).
2. Importare lo `.unitypackage` di SMPL-X for Unity
   (`Assets > Import Package > Custom Package...`).
3. Installare Newtonsoft Json dal Package Manager (vedi sopra).
4. Copiare `presets.json` dalla root del repository in
   `Assets/StreamingAssets/presets.json` (creare la cartella se manca).
   **Rifare la copia ogni volta che si rilancia `build_presets.py`**: è lo
   stesso file usato dall'app Python, con dentro sia le betas SMPL sia
   quelle SMPL-X (`betas_smplx`, quelle usate qui).
5. Trascinare in scena i prefab SMPL-X **maschile** e **femminile** dal
   pacchetto MPI (es. `SMPLX/Models/smplx-male` e `smplx-female`),
   posizionarli nell'origine.
6. Creare un GameObject vuoto `AvatarManager` e aggiungergli i componenti
   `AvatarController`, `UIManager` e `FitServiceClient`.
7. In `AvatarController` assegnare i riferimenti `maleAvatar`/`femaleAvatar`
   (il componente `SMPLX` dei due prefab in scena).
8. Costruire la Canvas uGUI. Ci sono due modi:
   - **Automatico (consigliato)**: menu Editor `Avatar SMPL > Build Settings
     Canvas` (script `Assets/Editor/BuildAvatarUI.cs`). Crea una `Canvas` con
     un pannello scorrevole a sinistra contenente tutti i widget elencati
     sotto, già dimensionati e organizzati in righe etichettate. Se in scena
     esiste già il GameObject `AvatarManager` con `UIManager` (punto 6), lo
     script collega **anche automaticamente** tutti i riferimenti
     dell'Inspector descritti al punto 9 (incluse le 9 chiavi di
     `Measurement Fields`), quindi conviene fare prima i punti 6-7 e poi
     lanciare questo comando. Rilanciabile: ricrea la Canvas da zero ogni
     volta (comodo se si vuole ripartire o modificare lo stile a mano dopo).
   - **Manuale**: creare a mano i widget seguenti dentro una `Canvas`:
     - un `Dropdown` per i preset;
     - un `Text` per la descrizione del preset;
     - un `Dropdown` per il genere (Male/Female);
     - **9 `InputField` per le misure target in cm** (il flusso principale,
       come nel main): altezza, circonferenze di torace/vita/fianchi/collo/
       testa, interno gamba, lunghezza braccio, larghezza spalle;
     - un `Button` **"Fit to Measurements"**;
     - un `Text` per le **Current Measurements**;
     - 10 `Slider` (min -5, max +5) per le betas come controllo avanzato,
       con accanto un `Text` opzionale per il valore;
     - un `Button` "Reset";
     - un `InputField` + `Button` "Load JSON" per caricare i parametri
       esportati dall'app Python;
     - un `Text` per i messaggi di stato.
9. In `UIManager` assegnare dall'Inspector tutti i riferimenti ai widget,
   all'`AvatarController` e al `FitServiceClient`. Per ogni elemento della
   lista **Measurement Fields** impostare la chiave testuale della misura
   (esattamente una tra: `height`, `chest circumference`,
   `waist circumference`, `hip circumference`, `inside leg height`,
   `arm right length`, `neck circumference`, `head circumference`,
   `shoulder breadth`) e il relativo `InputField`.
   **Se si è usato il comando automatico del punto 8**, questo passo è già
   fatto per i widget UI: restano solo da assegnare a mano `avatarController`
   e `fitClient` (riferimenti ad altri componenti, non a widget creati dallo
   script).
10. Avviare `python avatar_server.py` nella root del repo, poi Play:
    - selezionando un preset il corpo cambia istantaneamente e i campi
      misura si riempiono con i target del profilo;
    - modificando le misure e premendo **Fit to Measurements** il server
      esegue il fitting reale (qualche decina di secondi) e il corpo si
      aggiorna, con l'errore residuo massimo mostrato nello stato;
    - gli slider betas agiscono in tempo reale e il pannello Current
      Measurements si riaggiorna (calcolato dal misuratore geometrico vero).

## Interscambio con l'app Python

- **Preset**: entrambe le app leggono lo stesso `presets.json`. Le betas dei
  preset sono precalcolate da `build_presets.py` con il fitting sulle misure
  antropometriche target, in due varianti: `betas` (modello SMPL, usato
  dalla GUI Python) e `betas_smplx` (modello SMPL-X, usato qui). Servono due
  fit separati perché gli spazi di forma di SMPL e SMPL-X non sono compatibili.
- **Fitting**: il bottone "Fit to Measurements" usa il server locale, quindi
  misure e betas sono calcolate ESATTAMENTE come nella GUI Python (stesse
  funzioni di `fitting.py` e `measure.py`), sul modello SMPL-X del genere
  selezionato.
- **Avatar personalizzati**: nell'app Python `File > Save Model Params`
  salva, accanto al file joblib, un `.json` con genere e betas. Incollandone
  il percorso nell'`InputField` e premendo "Load JSON" si ricrea l'avatar in
  Unity. Per una corrispondenza fedele impostare il Body Model su **SMPLX**
  nell'app Python prima del fit e dell'export (con betas SMPL la forma su
  SMPL-X è solo approssimata; la UI mostra un avviso).

## Come funziona il tempo reale

Il pacchetto MPI espone in `SMPLX.cs` l'array `betas[10]`: ogni beta pilota
un blendshape della mesh (peso = beta x 100) e `UpdateJointPositions()`
ricalcola lo scheletro per la nuova corporatura. I nostri script si limitano
a scrivere le betas e chiamare questi metodi, quindi ogni aggiornamento
(preset, fit dal server, slider) modifica la mesh nello stesso frame; solo
il calcolo del fit avviene fuori da Unity, nel server Python.
