// ============================================================================
// AvatarController.cs
// Ponte tra la nostra UI e il componente SMPLX del pacchetto ufficiale
// "SMPL-X for Unity" (https://smpl-x.is.tue.mpg.de, script SMPLX.cs).
//
// Gestisce i due prefab (maschile e femminile) messi in scena: attiva quello
// del genere richiesto, scrive le 10 betas nel componente SMPLX e chiama i
// metodi del pacchetto che aggiornano blendshapes e scheletro:
//   - SetBetaShapes()       -> peso dei blendshapes "ShapeNNN" = beta * 100
//   - UpdateJointPositions() -> ricalcola le posizioni dei giunti per la nuova forma
//   - SnapToGroundPlane()   -> riappoggia i piedi al pavimento (y=0)
// ============================================================================
using UnityEngine;

public class AvatarController : MonoBehaviour
{
    public const int NUM_BETAS = 10;

    [Header("Prefab SMPL-X in scena (dal pacchetto SMPL-X for Unity)")]
    public SMPLX maleAvatar;
    public SMPLX femaleAvatar;

    private SMPLX _current;
    private string _currentGender = "male";

    void Start()
    {
        // All'avvio mostriamo il corpo maschile "medio" (betas a zero)
        if (_current == null)
            SetGender("male");
    }

    // Ritorna il componente SMPLX attivo (null se la scena non è configurata)
    public SMPLX Current => _current;

    // Genere attualmente mostrato ("male"/"female"), usato ad esempio dal
    // client di fitting per chiedere il fit sul modello del genere giusto.
    public string CurrentGender => _currentGender;

    // Attiva il prefab del genere richiesto ("male"/"female") e nasconde l'altro.
    public void SetGender(string gender)
    {
        bool isFemale = gender != null && gender.ToLower().StartsWith("f");
        _currentGender = isFemale ? "female" : "male";
        _current = isFemale ? femaleAvatar : maleAvatar;

        if (maleAvatar != null) maleAvatar.gameObject.SetActive(!isFemale);
        if (femaleAvatar != null) femaleAvatar.gameObject.SetActive(isFemale);
    }

    // Applica un intero vettore di betas all'avatar corrente.
    public void SetBetas(System.Collections.Generic.IList<float> betas)
    {
        if (_current == null || betas == null) return;
        for (int i = 0; i < NUM_BETAS && i < betas.Count; i++)
            _current.betas[i] = betas[i];
        RefreshShape();
    }

    // Modifica una singola beta (usato dagli slider della UI).
    public void SetBeta(int index, float value)
    {
        if (_current == null || index < 0 || index >= NUM_BETAS) return;
        _current.betas[index] = value;
        RefreshShape();
    }

    public float GetBeta(int index)
    {
        if (_current == null || index < 0 || index >= NUM_BETAS) return 0f;
        return _current.betas[index];
    }

    // Riporta l'avatar corrente al corpo "medio" (tutte le betas a zero).
    public void ResetShape()
    {
        if (_current == null) return;
        for (int i = 0; i < NUM_BETAS; i++)
            _current.betas[i] = 0f;
        RefreshShape();
    }

    // Propaga le betas correnti alla mesh: blendshapes + scheletro + appoggio a terra.
    private void RefreshShape()
    {
        _current.SetBetaShapes();
        _current.UpdateJointPositions();
        _current.SnapToGroundPlane();
    }
}
