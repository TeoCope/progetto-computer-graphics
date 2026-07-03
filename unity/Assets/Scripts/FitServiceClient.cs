// ============================================================================
// FitServiceClient.cs
// Client HTTP del server locale di fitting (avatar_server.py). Il server
// espone la STESSA logica della GUI Python (misuratore geometrico MeasureBody
// + ottimizzatore L-BFGS-B di fitting.py): Unity invia le misure in cm e
// riceve le betas, senza duplicare la logica in C#.
//
// Prerequisito: avviare nella root del repository
//     python avatar_server.py
// prima di premere Play (senza server restano comunque funzionanti i preset
// e gli slider, che non richiedono il fitting).
// ============================================================================
using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;
using Newtonsoft.Json;

[Serializable]
public class FitResult
{
    public List<float> betas;
    public Dictionary<string, float> achieved_measurements;
    public float max_error_cm;
    public float loss;
}

[Serializable]
public class MeasureResult
{
    public Dictionary<string, float> measurements;
}

[Serializable]
public class ServiceError
{
    public string error;
}

public class FitServiceClient : MonoBehaviour
{
    [Tooltip("Indirizzo del server avviato con: python avatar_server.py")]
    public string baseUrl = "http://localhost:8077";

    // Il corpo in scena è SMPL-X (pacchetto MPI), quindi il fitting va fatto
    // nello spazio di forma SMPL-X: gli spazi di SMPL e SMPL-X non sono compatibili.
    private const string MODEL_TYPE = "smplx";

    // Verifica che il server sia raggiungibile (da chiamare allo Start).
    public void CheckHealth(Action onOk, Action<string> onError)
    {
        StartCoroutine(GetCoroutine("/health", _ => onOk(), onError));
    }

    // Chiede al server le betas che riproducono le misure date (in cm).
    // measurements: {chiave misura -> cm}, le stesse chiavi della GUI Python.
    public void Fit(string gender, Dictionary<string, float> measurements,
                    Action<FitResult> onDone, Action<string> onError)
    {
        var payload = new Dictionary<string, object>
        {
            { "model_type", MODEL_TYPE },
            { "gender", gender },
            { "measurements", measurements },
        };
        StartCoroutine(PostCoroutine("/fit", payload,
            json => onDone(JsonConvert.DeserializeObject<FitResult>(json)), onError));
    }

    // Chiede al server le misure reali (in cm) del corpo con le betas date:
    // è ciò che alimenta il pannello "Current Measurements", come nel main.
    public void Measure(string gender, IList<float> betas,
                        Action<MeasureResult> onDone, Action<string> onError)
    {
        var payload = new Dictionary<string, object>
        {
            { "model_type", MODEL_TYPE },
            { "gender", gender },
            { "betas", betas },
        };
        StartCoroutine(PostCoroutine("/measure", payload,
            json => onDone(JsonConvert.DeserializeObject<MeasureResult>(json)), onError));
    }

    // ------- Coroutine HTTP -------

    private IEnumerator GetCoroutine(string path, Action<string> onDone, Action<string> onError)
    {
        using (var req = UnityWebRequest.Get(baseUrl + path))
        {
            req.timeout = 10;
            yield return req.SendWebRequest();
            HandleResponse(req, onDone, onError);
        }
    }

    private IEnumerator PostCoroutine(string path, object payload,
                                      Action<string> onDone, Action<string> onError)
    {
        byte[] body = Encoding.UTF8.GetBytes(JsonConvert.SerializeObject(payload));
        using (var req = new UnityWebRequest(baseUrl + path, "POST"))
        {
            req.uploadHandler = new UploadHandlerRaw(body);
            req.downloadHandler = new DownloadHandlerBuffer();
            req.SetRequestHeader("Content-Type", "application/json");
            // Il fit L-BFGS-B può richiedere decine di secondi: timeout generoso
            req.timeout = 180;
            yield return req.SendWebRequest();
            HandleResponse(req, onDone, onError);
        }
    }

    private void HandleResponse(UnityWebRequest req, Action<string> onDone, Action<string> onError)
    {
        if (req.result == UnityWebRequest.Result.ConnectionError)
        {
            onError("Server di fitting non raggiungibile: avviare 'python avatar_server.py' nella root del repo");
            return;
        }
        if (req.result != UnityWebRequest.Result.Success)
        {
            // Il server risponde agli errori con {"error": "..."}: lo mostriamo com'è
            string msg = req.downloadHandler != null ? req.downloadHandler.text : req.error;
            try
            {
                var err = JsonConvert.DeserializeObject<ServiceError>(msg);
                if (err != null && !string.IsNullOrEmpty(err.error)) msg = err.error;
            }
            catch { /* corpo non-JSON: teniamo il testo grezzo */ }
            onError(msg);
            return;
        }
        onDone(req.downloadHandler.text);
    }
}
