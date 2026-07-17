// ============================================================================
// UIManager.cs
// Interfaccia utente Unity (uGUI) che riproduce quella dell'app Python.
// Come nel main, l'utente definisce l'avatar inserendo le MISURE IN CM:
//   - Dropdown per scegliere il preset di avatar (da presets.json condiviso)
//   - Dropdown per il genere (male/female)
//   - Campi di input per le misure target (le stesse 9 della GUI Python;
//     campi vuoti o a 0 = nessun vincolo)
//   - Bottone "Fit to Measurements": invia le misure al server Python locale
//     (avatar_server.py), che esegue il fitting L-BFGS-B REALE e restituisce
//     le betas applicate al corpo in scena
//   - Pannello "Current Measurements" con le misure reali del corpo corrente
//     (calcolate dal misuratore geometrico del server)
//   - 10 slider per le betas come controllo "avanzato"
//   - Caricamento di un JSON di parametri esportato dall'app Python
// I riferimenti ai widget si assegnano dall'Inspector (vedi unity/README.md).
// ============================================================================
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

// Coppia (chiave misura, campo di input) assegnabile dall'Inspector: la
// chiave deve essere una delle 9 supportate dal server (es. "height",
// "waist circumference"...), il campo è l'InputField dove l'utente scrive i cm.
[System.Serializable]
public class MeasurementField
{
    public string key;
    public InputField field;
}

public class UIManager : MonoBehaviour
{
    [Header("Riferimenti")]
    public AvatarController avatarController;
    public FitServiceClient fitClient;

    [Header("Preset e genere")]
    public Dropdown presetDropdown;          // Voce 0 = "Custom", poi i preset
    public Text presetDescriptionText;       // Descrizione del preset selezionato
    public Dropdown genderDropdown;          // 0 = Male, 1 = Female
    public Dropdown textureDropdown;         // 0 = Nessuna, poi le texture trovate

    [Header("Misure target (cm)")]
    public List<MeasurementField> measurementFields;  // Le stesse 9 misure della GUI Python
    public Button fitButton;                 // "Fit to Measurements"
    public Text measurementsText;            // Pannello "Current Measurements"

    [Header("Controlli avanzati")]
    public List<Slider> betaSliders;         // 10 slider, range -5..+5
    public List<Text> betaValueLabels;       // Etichette col valore numerico (opzionali)
    public List<Slider> expressionSliders;   // 10 slider, range -2..+2
    public List<Text> expressionValueLabels;
    public Button resetButton;
    public InputField jsonPathInput;         // Percorso di un JSON esportato da Python
    public Button loadJsonButton;
    public Text statusText;                  // Messaggi di stato/avviso

    [Header("Animazioni (SUP)")]
    public Dropdown animationDropdown;       // Selezione dell'animazione
    public Button playAnimationButton;       // Bottone Play
    public SUPAnimationBridge animationBridge; // Riferimento al bridge

    private List<AvatarPreset> _presets = new List<AvatarPreset>();
    private List<string> _texturePaths = new List<string>();
    private List<string> _animationPaths = new List<string>(); // Percorsi dei file JSON
    private List<string> _animationNames = new List<string>(); // Nomi visualizzati (italiani)
    private bool _lastPlayingState = false;  // Ultimo stato noto del bridge, per aggiornare il bottone
    private bool _updatingUI = false;        // Evita che gli aggiornamenti programmatici rimettano "Custom"

    // Nomi visualizzati per le animazioni di esempio del package bmlSUP:
    // italiani e generici (non legati al genere dell'attore mocap).
    // Le animazioni non mappate ricadono sul nome del file.
    private static readonly Dictionary<string, string> AnimationDisplayNames = new Dictionary<string, string>
    {
        { "dancing_woman_1", "Danza 1" },
        { "dancing_woman_2", "Danza 2" },
        { "Talking_with_hands", "Conversazione" },
        { "woman_punching", "Pugilato" },
    };

    void Start()
    {
        _presets = AvatarPresetLoader.LoadPresets();

        // Popoliamo il dropdown dei preset: prima voce "Personalizzato"
        presetDropdown.ClearOptions();
        var options = new List<string> { "Personalizzato" };
        foreach (var p in _presets) options.Add(p.label);
        presetDropdown.AddOptions(options);
        presetDropdown.onValueChanged.AddListener(OnPresetSelected);

        if (genderDropdown != null)
        {
            genderDropdown.ClearOptions();
            genderDropdown.AddOptions(new List<string> { "Maschio", "Femmina" });
            genderDropdown.onValueChanged.AddListener(OnGenderSelected);
        }

        if (textureDropdown != null)
        {
            LoadTextureList();
            textureDropdown.onValueChanged.AddListener(OnTextureSelected);
        }

        // Colleghiamo gli slider delle betas (l'indice va "catturato" in una
        // variabile locale, altrimenti tutte le lambda vedrebbero l'ultimo i)
        if (betaSliders != null)
        {
            for (int i = 0; i < betaSliders.Count; i++)
            {
                int idx = i;
                betaSliders[i].minValue = -5f;
                betaSliders[i].maxValue = 5f;
                betaSliders[i].value = 0f;
                betaSliders[i].onValueChanged.AddListener(v => OnBetaSlider(idx, v));
            }
        }

        if (expressionSliders != null)
        {
            for (int i = 0; i < expressionSliders.Count; i++)
            {
                int idx = i;
                expressionSliders[i].minValue = -2f;
                expressionSliders[i].maxValue = 2f;
                expressionSliders[i].value = 0f;
                expressionSliders[i].onValueChanged.AddListener(v => OnExpressionSlider(idx, v));
            }
        }

        if (fitButton != null) fitButton.onClick.AddListener(OnFitToMeasurements);
        if (resetButton != null) resetButton.onClick.AddListener(OnReset);
        if (loadJsonButton != null) loadJsonButton.onClick.AddListener(OnLoadJson);

        if (animationDropdown != null && playAnimationButton != null)
        {
            LoadAnimationList();
            playAnimationButton.onClick.AddListener(OnPlayAnimation);
            UpdatePlayButtonVisual();
        }

        // Verifica che il server di fitting sia acceso (senza server restano
        // comunque utilizzabili preset e slider, che non richiedono il fit)
        if (fitClient != null)
            fitClient.CheckHealth(
                () => SetStatus($"Server di fitting online — {_presets.Count} preset caricati"),
                err => SetStatus(err));
        else
            SetStatus($"{_presets.Count} preset caricati (nessun FitServiceClient assegnato)");
    }

    // ------- Fitting dalle misure (flusso principale, come nel main) -------

    private void OnFitToMeasurements()
    {
        // Raccogliamo i campi compilati: come nella GUI Python, contano solo
        // le misure con valore > 0
        var targets = new Dictionary<string, float>();
        foreach (var mf in measurementFields)
        {
            if (mf.field == null || string.IsNullOrWhiteSpace(mf.field.text)) continue;
            if (float.TryParse(mf.field.text, System.Globalization.NumberStyles.Float,
                               System.Globalization.CultureInfo.InvariantCulture, out float v) && v > 0f)
                targets[mf.key] = v;
        }
        if (targets.Count == 0)
        {
            SetStatus("Inserire almeno una misura > 0 cm");
            return;
        }

        if (fitButton != null) fitButton.interactable = false;
        SetStatus("Fitting in corso sul server Python...");

        fitClient.Fit(avatarController.CurrentGender, targets,
            result =>
            {
                avatarController.SetBetas(result.betas);
                SyncSlidersFromAvatar();
                MarkCustom();
                ShowMeasurements(result.achieved_measurements);
                SetStatus($"Fit completato — errore max {result.max_error_cm:0.0} cm");
                if (fitButton != null) fitButton.interactable = true;
            },
            err =>
            {
                SetStatus(err);
                if (fitButton != null) fitButton.interactable = true;
            });
    }

    // Aggiorna il pannello "Current Measurements" chiedendo al server le
    // misure REALI del corpo corrente. Chiamato (con un piccolo debounce)
    // dopo ogni modifica degli slider.
    private void RefreshCurrentMeasurements()
    {
        if (fitClient == null) return;
        var betas = new List<float>();
        for (int i = 0; i < AvatarController.NUM_BETAS; i++)
            betas.Add(avatarController.GetBeta(i));
        fitClient.Measure(avatarController.CurrentGender, betas,
            result => ShowMeasurements(result.measurements),
            err => { /* server spento: il pannello semplicemente non si aggiorna */ });
    }

    private void ScheduleMeasurementsRefresh()
    {
        // Debounce: mentre l'utente trascina uno slider arrivano decine di
        // eventi al secondo; rimandiamo la richiesta finché non si ferma
        CancelInvoke(nameof(RefreshCurrentMeasurements));
        Invoke(nameof(RefreshCurrentMeasurements), 0.35f);
    }

    // ------- Callback UI -------

    private void OnPresetSelected(int index)
    {
        if (_updatingUI) return;
        if (index <= 0)
        {
            if (presetDescriptionText != null) presetDescriptionText.text = "";
            return;
        }

        var preset = _presets[index - 1];
        var betas = preset.GetUnityBetas();
        if (betas == null)
        {
            SetStatus($"Preset '{preset.label}' senza betas: esegui build_presets.py");
            return;
        }

        avatarController.SetGender(preset.gender);
        SyncGenderDropdown();
        avatarController.SetBetas(betas);
        SyncSlidersFromAvatar();

        // Precompiliamo i campi misura con i target del preset, come fa il
        // main: l'utente può partire da lì, modificarli e rifare il fit
        FillMeasurementFields(preset.target_measurements);

        if (presetDescriptionText != null)
            presetDescriptionText.text = preset.description;
        ShowMeasurements(preset.achieved_measurements_smplx ?? preset.target_measurements);
        SetStatus($"Preset '{preset.label}' applicato");
    }

    private void OnGenderSelected(int index)
    {
        if (_updatingUI) return;
        avatarController.SetGender(index == 1 ? "female" : "male");
        MarkCustom();
        ScheduleMeasurementsRefresh();
    }

    private void OnTextureSelected(int index)
    {
        if (_updatingUI) return;
        if (index <= 0)
        {
            avatarController.SetTexture(null);
        }
        else
        {
            string path = _texturePaths[index - 1];
            avatarController.SetTexture(path);
        }
    }

    private void OnBetaSlider(int index, float value)
    {
        if (_updatingUI) return;
        avatarController.SetBeta(index, value);
        UpdateBetaLabel(index, value);
        MarkCustom();
        ScheduleMeasurementsRefresh();
    }

    private void OnExpressionSlider(int index, float value)
    {
        if (_updatingUI) return;
        avatarController.SetExpression(index, value);
        UpdateExpressionLabel(index, value);
        MarkCustom();
    }

    private void OnReset()
    {
        avatarController.ResetShape();
        avatarController.ResetExpressions();
        SyncSlidersFromAvatar();
        MarkCustom();
        FillMeasurementFields(null);  // Svuota i campi misura
        if (measurementsText != null) measurementsText.text = "";
        SetStatus("Avatar riportato al corpo medio (e espressioni azzerate)");
        ScheduleMeasurementsRefresh();
    }

    private void OnLoadJson()
    {
        if (jsonPathInput == null || string.IsNullOrEmpty(jsonPathInput.text))
        {
            SetStatus("Inserisci il percorso di un file JSON esportato dall'app Python");
            return;
        }

        var p = AvatarPresetLoader.LoadAvatarParams(jsonPathInput.text.Trim());
        if (p == null)
        {
            SetStatus("File non trovato o non leggibile");
            return;
        }

        if (p.model_type != null && p.model_type.ToLower() == "smpl")
        {
            // Le betas SMPL non sono trasferibili 1:1 su SMPL-X: avvisiamo
            // (per un interscambio fedele, nell'app Python usare il modello SMPLX)
            SetStatus("Attenzione: il file usa betas SMPL, la forma su SMPL-X sara' approssimata");
        }
        else
        {
            SetStatus($"Avatar '{p.preset_id}' caricato dal JSON");
        }

        avatarController.SetGender(p.gender);
        SyncGenderDropdown();
        avatarController.SetBetas(p.betas);
        SyncSlidersFromAvatar();
        MarkCustom();
        FillMeasurementFields(p.measurements);
        ScheduleMeasurementsRefresh();
    }

    private void OnPlayAnimation()
    {
        if (animationBridge == null)
        {
            SetStatus("Nessun SUPAnimationBridge assegnato allo UIManager!");
            return;
        }

        // Il bottone è un toggle: se sta riproducendo, ferma; altrimenti avvia
        // l'animazione selezionata nel dropdown.
        if (animationBridge.isPlaying)
        {
            animationBridge.StopAnimation();
            SetStatus("Animazione fermata");
        }
        else if (_animationPaths.Count == 0)
        {
            SetStatus("Nessuna animazione trovata");
        }
        else
        {
            animationBridge.LoadAnimation(_animationPaths[animationDropdown.value]);
            SetStatus($"Riproduzione animazione: {_animationNames[animationDropdown.value]}");
        }

        UpdatePlayButtonVisual();
    }

    // Aggiorna testo e colore del bottone Play/Stop in base allo stato del bridge:
    // verde "Riproduci" quando fermo, rosso "Ferma" durante la riproduzione.
    private void UpdatePlayButtonVisual()
    {
        if (playAnimationButton == null) return;

        bool playing = animationBridge != null && animationBridge.isPlaying;
        _lastPlayingState = playing;

        var image = playAnimationButton.GetComponent<Image>();
        if (image != null)
            image.color = playing ? new Color(0.8f, 0.25f, 0.25f) : new Color(0.2f, 0.6f, 0.3f);

        var label = playAnimationButton.GetComponentInChildren<Text>();
        if (label != null)
            label.text = playing ? "Ferma" : "Riproduci";
    }

    void Update()
    {
        // L'animazione può fermarsi da sola (loop disattivato): se lo stato del
        // bridge è cambiato rispetto all'ultimo noto, riallinea il bottone.
        if (animationBridge != null && animationBridge.isPlaying != _lastPlayingState)
            UpdatePlayButtonVisual();
    }

    // ------- Helper -------

    private void LoadTextureList()
    {
        textureDropdown.ClearOptions();
        var options = new List<string> { "Nessuna" };
        _texturePaths.Clear();

        string path = System.IO.Path.Combine(Application.dataPath, "../../data/textures");
        if (System.IO.Directory.Exists(path))
        {
            string[] files = System.IO.Directory.GetFiles(path);
            foreach (string file in files)
            {
                string ext = System.IO.Path.GetExtension(file).ToLower();
                if (ext == ".png" || ext == ".jpg" || ext == ".jpeg")
                {
                    options.Add(System.IO.Path.GetFileNameWithoutExtension(file));
                    _texturePaths.Add(file);
                }
            }
        }

        textureDropdown.AddOptions(options);
    }

    private void LoadAnimationList()
    {
        animationDropdown.ClearOptions();
        var options = new List<string>();
        _animationPaths.Clear();
        _animationNames.Clear();

        // Carica dalle sample animations del pacchetto bmlSUP
        string path = System.IO.Path.Combine(Application.dataPath, "../Packages/com.biomotionlab.sup/SampleAnimations/JsonAnimations");
        if (System.IO.Directory.Exists(path))
        {
            string[] files = System.IO.Directory.GetFiles(path, "*.json");
            foreach (string file in files)
            {
                string fileName = System.IO.Path.GetFileNameWithoutExtension(file);
                string displayName = AnimationDisplayNames.TryGetValue(fileName, out string mapped)
                    ? mapped : fileName;
                options.Add(displayName);
                _animationNames.Add(displayName);
                _animationPaths.Add(file);
            }
        }

        animationDropdown.AddOptions(options);

        // Selezione di default: "Danza 1" (se presente), poi sarà l'utente a cambiarla
        int defaultIndex = _animationNames.IndexOf("Danza 1");
        animationDropdown.SetValueWithoutNotify(defaultIndex >= 0 ? defaultIndex : 0);
        animationDropdown.RefreshShownValue();
    }

    // Riporta il dropdown dei preset su "Custom" (la forma corrente non
    // corrisponde più a un preset) senza far scattare la callback.
    private void MarkCustom()
    {
        _updatingUI = true;
        presetDropdown.value = 0;
        if (presetDescriptionText != null) presetDescriptionText.text = "";
        _updatingUI = false;
    }

    // Allinea gli slider (e le etichette) alle betas correnti dell'avatar,
    // senza far scattare le callback degli slider.
    private void SyncSlidersFromAvatar()
    {
        _updatingUI = true;
        if (betaSliders != null)
        {
            for (int i = 0; i < betaSliders.Count; i++)
            {
                float v = avatarController.GetBeta(i);
                betaSliders[i].value = v;
                UpdateBetaLabel(i, v);
            }
        }
        if (expressionSliders != null)
        {
            for (int i = 0; i < expressionSliders.Count; i++)
            {
                float v = avatarController.GetExpression(i);
                expressionSliders[i].value = v;
                UpdateExpressionLabel(i, v);
            }
        }
        _updatingUI = false;
    }

    private void SyncGenderDropdown()
    {
        if (genderDropdown == null) return;
        _updatingUI = true;
        genderDropdown.value = avatarController.CurrentGender == "female" ? 1 : 0;
        _updatingUI = false;
    }

    // Riempie i campi misura con i valori dati (null o chiave assente = campo vuoto)
    private void FillMeasurementFields(Dictionary<string, float> values)
    {
        foreach (var mf in measurementFields)
        {
            if (mf.field == null) continue;
            if (values != null && values.TryGetValue(mf.key, out float v) && v > 0f)
                mf.field.text = v.ToString("0.0", System.Globalization.CultureInfo.InvariantCulture);
            else
                mf.field.text = "";
        }
    }

    private void UpdateBetaLabel(int index, float value)
    {
        if (betaValueLabels != null && index < betaValueLabels.Count && betaValueLabels[index] != null)
            betaValueLabels[index].text = value.ToString("0.0");
    }

    private void UpdateExpressionLabel(int index, float value)
    {
        if (expressionValueLabels != null && index < expressionValueLabels.Count && expressionValueLabels[index] != null)
            expressionValueLabels[index].text = value.ToString("0.0");
    }

    private void ShowMeasurements(Dictionary<string, float> measurements)
    {
        if (measurementsText == null || measurements == null) return;
        
        var keyToLabel = new Dictionary<string, string>
        {
            { "height", "Altezza" },
            { "chest circumference", "Circonf. torace" },
            { "waist circumference", "Circonf. vita" },
            { "hip circumference", "Circonf. fianchi" },
            { "neck circumference", "Circonf. collo" },
            { "head circumference", "Circonf. testa" },
            { "inside leg height", "Interno gamba" },
            { "arm right length", "Lunghezza braccio" },
            { "shoulder breadth", "Larghezza spalle" }
        };

        var sb = new System.Text.StringBuilder();
        foreach (var kv in measurements)
        {
            string label = keyToLabel.TryGetValue(kv.Key, out string itLabel) ? itLabel : kv.Key;
            sb.AppendLine($"{label}: {kv.Value:0.0} cm");
        }
        measurementsText.text = sb.ToString();
    }

    private void SetStatus(string msg)
    {
        if (statusText != null) statusText.text = msg;
        Debug.Log(msg);
    }
}
