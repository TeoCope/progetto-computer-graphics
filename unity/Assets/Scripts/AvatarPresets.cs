// ============================================================================
// AvatarPresets.cs
// Classi che rispecchiano lo schema di presets.json (lo stesso file usato
// dall'app Python) e caricamento del file da StreamingAssets.
//
// NOTA: usiamo Newtonsoft Json (pacchetto "com.unity.nuget.newtonsoft-json",
// vedi README) perché JsonUtility di Unity non supporta i dizionari, e le
// misure antropometriche sono salvate come dizionario {nome misura: cm}.
// ============================================================================
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using Newtonsoft.Json;

[System.Serializable]
public class AvatarPreset
{
    public string id;
    public string label;
    public string description;
    public string model_type;
    public string gender;
    public Dictionary<string, float> target_measurements;

    // Betas per il modello SMPL (usate dall'app Python con model_type=smpl)
    public List<float> betas;
    public Dictionary<string, float> achieved_measurements;

    // Betas per SMPL-X: sono quelle da usare in Unity, perché il pacchetto
    // "SMPL-X for Unity" usa il modello SMPL-X e gli spazi di forma di SMPL
    // e SMPL-X non sono compatibili tra loro.
    public List<float> betas_smplx;
    public Dictionary<string, float> achieved_measurements_smplx;

    // Ritorna le betas adatte a Unity: preferisce betas_smplx, altrimenti
    // ripiega su betas (corrette solo se il preset è già di tipo smplx).
    public List<float> GetUnityBetas()
    {
        if (betas_smplx != null && betas_smplx.Count > 0) return betas_smplx;
        return betas;
    }
}

[System.Serializable]
public class AvatarPresetFile
{
    public List<AvatarPreset> presets;
}

// Parametri di un singolo avatar esportati dall'app Python con
// "File > Save Model Params" (file .json accanto al .joblib).
[System.Serializable]
public class AvatarParams
{
    public string model_type;
    public string gender;
    public string preset_id;
    public List<float> betas;
    public List<float> expression;
    public Dictionary<string, float> measurements;
}

public static class AvatarPresetLoader
{
    // Carica presets.json da StreamingAssets (copiato lì dalla root del
    // repository: è la stessa identica fonte dati dell'app Python).
    public static List<AvatarPreset> LoadPresets(string fileName = "presets.json")
    {
        string path = Path.Combine(Application.streamingAssetsPath, fileName);
        if (!File.Exists(path))
        {
            Debug.LogWarning($"Presets file not found: {path}");
            return new List<AvatarPreset>();
        }
        string json = File.ReadAllText(path);
        var file = JsonConvert.DeserializeObject<AvatarPresetFile>(json);
        return file?.presets ?? new List<AvatarPreset>();
    }

    // Carica un file di parametri avatar esportato dall'app Python.
    public static AvatarParams LoadAvatarParams(string absolutePath)
    {
        if (!File.Exists(absolutePath))
        {
            Debug.LogWarning($"Avatar params file not found: {absolutePath}");
            return null;
        }
        string json = File.ReadAllText(absolutePath);
        return JsonConvert.DeserializeObject<AvatarParams>(json);
    }
}
