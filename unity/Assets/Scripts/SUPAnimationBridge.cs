using UnityEngine;
using System.IO;
using ThirdParty.SimpleJSON; // Fornito dal package bmlSUP
using SMPLModel; // Namespace di bmlSUP che definisce le ossa e utilities

public class SUPAnimationBridge : MonoBehaviour
{
    [Header("Riferimento all'Avatar Controller")]
    public AvatarController avatarController;
    
    // Riferimento interno all'avatar attualmente attivo
    private SMPLX currentAvatar;

    [Header("Configurazione Animazione")]
    public string amassJsonPath;
    public float playbackSpeed = 1f;
    public bool loop = true;

    // Proprietà (non serializzata): all'avvio è sempre false, anche se la scena
    // è stata salvata durante una riproduzione.
    public bool isPlaying { get; private set; }

    private int frameCount = 0;
    private int currentFrame = 0;
    private float fps = 60f;
    private float timer = 0f;

    private Quaternion[,] poses;
    private Vector3[] translations;
    private Transform[] avatarBones;

    // Pelvis dell'avatar corrente e sua posizione locale di riposo: le traslazioni
    // dell'animazione vengono applicate come delta rispetto a questa, così il
    // frame 0 coincide esattamente con l'altezza della posa di riposo.
    private Transform pelvisBone;
    private Vector3 pelvisRestLocalPosition;
    
    // Correzione dinamica calcolata al frame 0 per girare l'avatar in avanti
    private Quaternion yCorrection = Quaternion.identity;

    // Struttura base di AMASS/SMPL (52 joint incluso le mani)
    public const int NUM_JOINTS = 52;

    // Mappatura nomi ossa SMPL-X (tutti minuscoli) -> Indici AMASS/SMPL-H
    private static readonly System.Collections.Generic.Dictionary<string, int> smplxToSupIndex = new System.Collections.Generic.Dictionary<string, int> {
        {"pelvis", 0}, {"left_hip", 1}, {"right_hip", 2}, {"spine1", 3}, {"left_knee", 4}, {"right_knee", 5}, {"spine2", 6}, {"left_ankle", 7}, {"right_ankle", 8}, {"spine3", 9}, {"left_foot", 10}, {"right_foot", 11}, {"neck", 12}, {"left_collar", 13}, {"right_collar", 14}, {"head", 15}, {"left_shoulder", 16}, {"right_shoulder", 17}, {"left_elbow", 18}, {"right_elbow", 19}, {"left_wrist", 20}, {"right_wrist", 21},
        // Mano sinistra
        {"left_index1", 22}, {"left_index2", 23}, {"left_index3", 24}, {"left_middle1", 25}, {"left_middle2", 26}, {"left_middle3", 27}, {"left_pinky1", 28}, {"left_pinky2", 29}, {"left_pinky3", 30}, {"left_ring1", 31}, {"left_ring2", 32}, {"left_ring3", 33}, {"left_thumb1", 34}, {"left_thumb2", 35}, {"left_thumb3", 36},
        // Mano destra
        {"right_index1", 37}, {"right_index2", 38}, {"right_index3", 39}, {"right_middle1", 40}, {"right_middle2", 41}, {"right_middle3", 42}, {"right_pinky1", 43}, {"right_pinky2", 44}, {"right_pinky3", 45}, {"right_ring1", 46}, {"right_ring2", 47}, {"right_ring3", 48}, {"right_thumb1", 49}, {"right_thumb2", 50}, {"right_thumb3", 51}
    };

    void Start()
    {
        // Non avviare l'animazione in automatico
        // Viene gestita tramite UIManager.cs
    }

    public void StopAnimation()
    {
        isPlaying = false;
        
        // Opzionale: rimetti il personaggio nella posa "T" o iniziale se si vuole
        if (avatarBones != null)
        {
            for (int i = 0; i < avatarBones.Length; i++)
            {
                avatarBones[i].localEulerAngles = Vector3.zero;
            }
        }
        
        if (pelvisBone != null) pelvisBone.localPosition = pelvisRestLocalPosition;
    }

    public void LoadAnimation(string path)
    {
        if (!File.Exists(path))
        {
            Debug.LogError("File JSON di animazione non trovato: " + path);
            return;
        }

        string jsonString = File.ReadAllText(path);
        JSONNode root = JSON.Parse(jsonString);

        if (root["mocap_framerate"] != null)
            fps = root["mocap_framerate"].AsFloat;
        else
            fps = 60f;

        JSONNode transNode = root["trans"];
        JSONNode posesNode = root["poses"];

        frameCount = transNode.Count;
        translations = new Vector3[frameCount];
        poses = new Quaternion[frameCount, NUM_JOINTS];

        Vector3 initialOffset = Vector3.zero;
        yCorrection = Quaternion.identity;

        for (int f = 0; f < frameCount; f++)
        {
            float tx = transNode[f][0].AsFloat;
            float ty = transNode[f][1].AsFloat;
            float tz = transNode[f][2].AsFloat;
            // Conversione coordinare SMPL (Right-Handed) -> Unity (Left-Handed per SMPL-X)
            // SMPL-X usa: inversione dell'asse X per le posizioni
            Vector3 rawTrans = new Vector3(-tx, ty, tz);

            for (int j = 0; j < NUM_JOINTS; j++)
            {
                if (j >= posesNode[f].Count) break;

                float x = posesNode[f][j][0].AsFloat;
                float y = posesNode[f][j][1].AsFloat;
                float z = posesNode[f][j][2].AsFloat;
                float w = posesNode[f][j][3].AsFloat;

                Quaternion smplRot = new Quaternion(x, y, z, w);
                // Conversione Right-Handed -> Left-Handed basata sulla convenzione di SMPL-X
                poses[f, j] = new Quaternion(smplRot.x, -smplRot.y, -smplRot.z, smplRot.w);
            }

            if (f == 0)
            {
                // Salviamo l'offset iniziale completo (X, Y e Z): la Y del mocap è
                // l'altezza assoluta del bacino dell'attore e va sottratta, altrimenti
                // l'avatar viene sollevato rispetto alla posa di riposo (pelvis a zero).
                // Il movimento verticale relativo (salti, piegamenti) resta preservato.
                initialOffset = rawTrans;
                
                // Calcoliamo la direzione in cui sta guardando il bacino al primo frame
                Quaternion initialPelvisRot = Quaternion.Euler(-90, 0, 0) * poses[0, 0];
                Vector3 fwd = initialPelvisRot * Vector3.forward;
                fwd.y = 0; // Ignoriamo la pendenza (es. se è piegato in avanti)
                if (fwd.sqrMagnitude > 0.001f)
                {
                    fwd.Normalize();
                    // Calcoliamo la rotazione necessaria per allinearlo verso Vector3.forward (+Z locale)
                    float angle = Vector3.SignedAngle(fwd, Vector3.forward, Vector3.up);
                    yCorrection = Quaternion.Euler(0, angle, 0);
                }
            }

            // Applichiamo la correzione alla traslazione, così se l'attore camminava in avanti 
            // nella stanza, camminerà in avanti anche nella nuova direzione corretta!
            translations[f] = yCorrection * (rawTrans - initialOffset);
        }

        // L'assegnazione delle ossa ora avviene in modo dinamico in Update()

        currentFrame = 0;
        isPlaying = true;
    }

    void LateUpdate()
    {
        // Aggiorna dinamicamente l'avatar corrente se l'AvatarController cambia (es. cambio genere)
        if (avatarController != null && avatarController.Current != currentAvatar)
        {
            currentAvatar = avatarController.Current;
            if (currentAvatar != null)
            {
                SkinnedMeshRenderer smr = currentAvatar.GetComponentInChildren<SkinnedMeshRenderer>();
                if (smr != null)
                    avatarBones = smr.bones;
                else
                    avatarBones = null;

                // Catturiamo il pelvis e la sua posizione di riposo del nuovo avatar
                pelvisBone = GetBoneByName("pelvis");
                pelvisRestLocalPosition = pelvisBone != null ? pelvisBone.localPosition : Vector3.zero;
            }
            else
            {
                avatarBones = null;
                pelvisBone = null;
            }
        }

        if (!isPlaying || frameCount == 0 || currentAvatar == null || avatarBones == null) return;

        timer += Time.deltaTime * playbackSpeed;
        float frameDuration = 1f / fps;

        while (timer >= frameDuration)
        {
            timer -= frameDuration;
            currentFrame++;
            if (currentFrame >= frameCount)
            {
                if (loop) currentFrame = 0;
                else
                {
                    currentFrame = frameCount - 1;
                    isPlaying = false;
                }
            }
        }

        ApplyPose(currentFrame);
    }

    void ApplyPose(int frame)
    {
        // 1. Applica la traslazione al pelvis come delta rispetto alla posa di riposo
        if (pelvisBone != null)
        {
            pelvisBone.localPosition = pelvisRestLocalPosition + translations[frame];
        }

        // 2. Applica rotazioni
        for (int i = 0; i < avatarBones.Length; i++)
        {
            string bName = avatarBones[i].name;
            
            // Verifichiamo se l'osso SMPL-X ha una corrispondenza nel dizionario AMASS
            if (smplxToSupIndex.TryGetValue(bName, out int poseIdx))
            {
                if (poseIdx < NUM_JOINTS)
                {
                    // Resetta la rotazione e applica l'animazione
                    avatarBones[i].localEulerAngles = Vector3.zero;
                    
                    // Applica la rotazione locale
                    if (bName == "pelvis")
                    {
                        // -90 su X lo alza in piedi. yCorrection ruota dinamicamente l'animazione verso +Z (di fronte).
                        avatarBones[i].localRotation = yCorrection * Quaternion.Euler(-90, 0, 0) * poses[frame, poseIdx];
                    }
                    else
                    {
                        avatarBones[i].localRotation = poses[frame, poseIdx];
                    }
                }
            }
        }
    }

    Transform GetBoneByName(string n)
    {
        if (avatarBones == null) return null;
        foreach (var b in avatarBones)
        {
            if (b.name == n) return b;
        }
        return null;
    }
}
