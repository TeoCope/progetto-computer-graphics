// ============================================================================
// BuildAvatarUI.cs (Editor-only)
// Genera automaticamente la Canvas uGUI descritta al punto 8 di unity/README.md
// (dropdown preset/genere, 9 InputField per le misure, bottone Fit, pannello
// Current Measurements, 10 slider betas, Reset, Load JSON, status text).
// Se in scena esiste già un UIManager (punto 6), collega anche tutti i
// riferimenti dell'Inspector (punto 9), evitando il drag-and-drop manuale.
//
// Uso: menu "Avatar SMPL > Build Settings Canvas".
// ============================================================================
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

public static class BuildAvatarUI
{
    private static readonly string[] MeasurementKeys =
    {
        "height", "chest circumference", "waist circumference", "hip circumference",
        "neck circumference", "head circumference", "inside leg height",
        "arm right length", "shoulder breadth",
    };

    private static readonly string[] MeasurementLabels =
    {
        "Altezza", "Circonf. torace", "Circonf. vita", "Circonf. fianchi",
        "Circonf. collo", "Circonf. testa", "Interno gamba",
        "Lunghezza braccio", "Larghezza spalle",
    };

    // Nomi intuitivi (approssimati in base alle componenti principali di FLAME/SMPL-X)
    private static readonly string[] ExpressionLabels =
    {
        "Apertura Bocca",
        "Sorriso / Broncio",
        "Labbra in fuori",
        "Sopracciglia",
        "Occhi / Sguardo",
        "Guance",
        "Corrucciato",
        "Smorfia",
        "Sorpresa",
        "Disgusto"
    };

    private static DefaultControls.Resources s_Resources;

    private static DefaultControls.Resources Resources
    {
        get
        {
            if (s_Resources.standard == null)
            {
                s_Resources.standard = AssetDatabase.GetBuiltinExtraResource<Sprite>("UI/Skin/UISprite.psd");
                s_Resources.background = AssetDatabase.GetBuiltinExtraResource<Sprite>("UI/Skin/Background.psd");
                s_Resources.inputField = AssetDatabase.GetBuiltinExtraResource<Sprite>("UI/Skin/InputFieldBackground.psd");
                s_Resources.knob = AssetDatabase.GetBuiltinExtraResource<Sprite>("UI/Skin/Knob.psd");
                s_Resources.checkmark = AssetDatabase.GetBuiltinExtraResource<Sprite>("UI/Skin/Checkmark.psd");
                s_Resources.dropdown = AssetDatabase.GetBuiltinExtraResource<Sprite>("UI/Skin/DropdownArrow.psd");
                s_Resources.mask = AssetDatabase.GetBuiltinExtraResource<Sprite>("UI/Skin/UIMask.psd");
            }
            return s_Resources;
        }
    }

    [MenuItem("Avatar SMPL/Build Settings Canvas")]
    public static void Build()
    {
        var existing = GameObject.Find("AvatarUICanvas");
        if (existing != null) Undo.DestroyObjectImmediate(existing);

        if (Object.FindObjectOfType<EventSystem>() == null)
        {
            var esGO = new GameObject("EventSystem", typeof(EventSystem), typeof(StandaloneInputModule));
            Undo.RegisterCreatedObjectUndo(esGO, "Create EventSystem");
        }

        var canvasGO = new GameObject("AvatarUICanvas", typeof(Canvas), typeof(CanvasScaler), typeof(GraphicRaycaster));
        Undo.RegisterCreatedObjectUndo(canvasGO, "Create AvatarUICanvas");
        canvasGO.GetComponent<Canvas>().renderMode = RenderMode.ScreenSpaceOverlay;
        var scaler = canvasGO.GetComponent<CanvasScaler>();
        scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
        scaler.referenceResolution = new Vector2(1080, 1080);

        var panelGO = DefaultControls.CreatePanel(Resources);
        panelGO.name = "SettingsPanel";
        panelGO.transform.SetParent(canvasGO.transform, false);
        var panelRT = panelGO.GetComponent<RectTransform>();
        panelRT.anchorMin = new Vector2(0f, 0f);
        panelRT.anchorMax = new Vector2(0.36f, 1f);
        panelRT.offsetMin = new Vector2(10, 10);
        panelRT.offsetMax = new Vector2(-5, -10);
        panelGO.GetComponent<Image>().color = new Color(0f, 0f, 0f, 0.6f);

        var scrollGO = DefaultControls.CreateScrollView(Resources);
        scrollGO.name = "SettingsScrollView";
        scrollGO.transform.SetParent(panelGO.transform, false);
        var scrollRT = scrollGO.GetComponent<RectTransform>();
        scrollRT.anchorMin = Vector2.zero;
        scrollRT.anchorMax = Vector2.one;
        scrollRT.offsetMin = new Vector2(6, 6);
        scrollRT.offsetMax = new Vector2(-6, -6);
        var scrollRect = scrollGO.GetComponent<ScrollRect>();
        scrollRect.horizontal = false;

        var content = scrollRect.content;
        content.anchorMin = new Vector2(0f, 1f);
        content.anchorMax = new Vector2(1f, 1f);
        content.pivot = new Vector2(0.5f, 1f);
        var vlg = content.gameObject.AddComponent<VerticalLayoutGroup>();
        vlg.childForceExpandWidth = true;
        vlg.childForceExpandHeight = false;
        vlg.childControlWidth = true;
        vlg.childControlHeight = true;
        vlg.spacing = 6;
        vlg.padding = new RectOffset(8, 8, 8, 8);
        content.gameObject.AddComponent<ContentSizeFitter>().verticalFit = ContentSizeFitter.FitMode.PreferredSize;

        // --- Preset e genere ---
        AddSectionHeader(content, "Preset");
        var presetGO = DefaultControls.CreateDropdown(Resources);
        FixUIControl(presetGO);
        var presetDropdown = AddFieldRow(content, "Preset", presetGO).GetComponent<Dropdown>();

        var presetDescriptionText = AddPlainText(content, "", 40);

        var genderGO = DefaultControls.CreateDropdown(Resources);
        FixUIControl(genderGO);
        var genderDropdown = AddFieldRow(content, "Genere", genderGO).GetComponent<Dropdown>();

        var textureGO = DefaultControls.CreateDropdown(Resources);
        FixUIControl(textureGO);
        var textureDropdown = AddFieldRow(content, "Texture", textureGO).GetComponent<Dropdown>();

        // --- Misure target ---
        AddSectionHeader(content, "Misure desiderate (cm)");
        var measurementInputFields = new List<InputField>();
        foreach (var label in MeasurementLabels)
        {
            var fieldGO = DefaultControls.CreateInputField(Resources);
            FixUIControl(fieldGO);
            var field = fieldGO.GetComponent<InputField>();
            field.contentType = InputField.ContentType.DecimalNumber;
            AddFieldRow(content, label, fieldGO);
            measurementInputFields.Add(field);
        }
        var fitButton = AddButton(content, "Adatta alle Misure", 32, new Color(0.2f, 0.6f, 0.2f), Color.white);

        // --- Misure attuali ---
        AddSectionHeader(content, "Misure attuali");
        var measurementsText = AddPlainText(content, "", 160);

        // --- Espressioni Facciali ---
        AddSectionHeader(content, "Espressioni Facciali");
        var expressionSliders = new List<Slider>();
        var expressionValueLabels = new List<Text>();
        for (int i = 0; i < 10; i++)
        {
            var sliderGO = DefaultControls.CreateSlider(Resources);
            var slider = sliderGO.GetComponent<Slider>();
            expressionSliders.Add(slider);
            var valueLabel = AddSliderRow(content, ExpressionLabels[i], sliderGO);
            expressionValueLabels.Add(valueLabel);
        }

        // --- Animazioni (SUP) ---
        AddSectionHeader(content, "Animazioni (SUP)");
        var animDropdownGO = DefaultControls.CreateDropdown(Resources);
        FixUIControl(animDropdownGO);
        var animationDropdown = AddFieldRow(content, "Seleziona", animDropdownGO).GetComponent<Dropdown>();
        
        // Stato iniziale "fermo": a runtime UIManager lo alterna con "Ferma"/rosso
        var playAnimationButton = AddButton(content, "Riproduci", 32, new Color(0.2f, 0.6f, 0.3f), Color.white);

        // --- Reset ---
        var resetButton = AddButton(content, "Reimposta", 32, new Color(0.8f, 0.3f, 0.3f), Color.white);

        // --- Le sezioni Betas e Carica JSON sono state rimosse ---
        var betaSliders = new List<Slider>();
        var betaValueLabels = new List<Text>();
        InputField jsonPathInput = null;
        Button loadJsonButton = null;

        // --- Stato ---
        AddSectionHeader(content, "Stato");
        var statusText = AddPlainText(content, "", 60);

        WireUIManager(presetDropdown, presetDescriptionText, genderDropdown, textureDropdown,
            measurementInputFields, fitButton, measurementsText,
            betaSliders, betaValueLabels, expressionSliders, expressionValueLabels, resetButton, jsonPathInput, loadJsonButton, statusText, animationDropdown, playAnimationButton);

        Selection.activeGameObject = canvasGO;
        EditorUtility.DisplayDialog("Avatar UI",
            "Canvas creata. Se in scena esiste già un GameObject con UIManager i riferimenti sono " +
            "stati collegati automaticamente; altrimenti crea prima 'AvatarManager' (punti 6-7 del " +
            "README) e rilancia questo comando per il collegamento automatico.", "OK");
    }

    private static void WireUIManager(Dropdown presetDropdown, Text presetDescriptionText, Dropdown genderDropdown, Dropdown textureDropdown,
        List<InputField> measurementInputFields, Button fitButton, Text measurementsText,
        List<Slider> betaSliders, List<Text> betaValueLabels, List<Slider> expressionSliders, List<Text> expressionValueLabels, Button resetButton,
        InputField jsonPathInput, Button loadJsonButton, Text statusText, Dropdown animationDropdown, Button playAnimationButton)
    {
        var uiManager = Object.FindObjectOfType<UIManager>();
        if (uiManager == null)
        {
            Debug.LogWarning("Nessun UIManager in scena: crea 'AvatarManager' con i componenti " +
                "AvatarController/UIManager/FitServiceClient e rilancia 'Avatar SMPL > Build Settings " +
                "Canvas' per il collegamento automatico dei riferimenti.");
            return;
        }

        var so = new SerializedObject(uiManager);
        so.FindProperty("presetDropdown").objectReferenceValue = presetDropdown;
        so.FindProperty("presetDescriptionText").objectReferenceValue = presetDescriptionText;
        so.FindProperty("genderDropdown").objectReferenceValue = genderDropdown;
        so.FindProperty("textureDropdown").objectReferenceValue = textureDropdown;

        var measFieldsProp = so.FindProperty("measurementFields");
        measFieldsProp.arraySize = MeasurementKeys.Length;
        for (int i = 0; i < MeasurementKeys.Length; i++)
        {
            var el = measFieldsProp.GetArrayElementAtIndex(i);
            el.FindPropertyRelative("key").stringValue = MeasurementKeys[i];
            el.FindPropertyRelative("field").objectReferenceValue = measurementInputFields[i];
        }

        so.FindProperty("fitButton").objectReferenceValue = fitButton;
        so.FindProperty("measurementsText").objectReferenceValue = measurementsText;

        var slidersProp = so.FindProperty("betaSliders");
        slidersProp.arraySize = betaSliders.Count;
        for (int i = 0; i < betaSliders.Count; i++)
            slidersProp.GetArrayElementAtIndex(i).objectReferenceValue = betaSliders[i];

        var labelsProp = so.FindProperty("betaValueLabels");
        labelsProp.arraySize = betaValueLabels.Count;
        for (int i = 0; i < betaValueLabels.Count; i++)
            labelsProp.GetArrayElementAtIndex(i).objectReferenceValue = betaValueLabels[i];

        var exprSlidersProp = so.FindProperty("expressionSliders");
        exprSlidersProp.arraySize = expressionSliders.Count;
        for (int i = 0; i < expressionSliders.Count; i++)
            exprSlidersProp.GetArrayElementAtIndex(i).objectReferenceValue = expressionSliders[i];

        var exprLabelsProp = so.FindProperty("expressionValueLabels");
        exprLabelsProp.arraySize = expressionValueLabels.Count;
        for (int i = 0; i < expressionValueLabels.Count; i++)
            exprLabelsProp.GetArrayElementAtIndex(i).objectReferenceValue = expressionValueLabels[i];

        so.FindProperty("resetButton").objectReferenceValue = resetButton;
        so.FindProperty("jsonPathInput").objectReferenceValue = jsonPathInput;
        so.FindProperty("loadJsonButton").objectReferenceValue = loadJsonButton;
        so.FindProperty("statusText").objectReferenceValue = statusText;

        so.FindProperty("animationDropdown").objectReferenceValue = animationDropdown;
        so.FindProperty("playAnimationButton").objectReferenceValue = playAnimationButton;
        
        // Collega in automatico anche l'Animation Bridge (cercandolo sullo stesso GameObject dell'UIManager)
        var bridge = uiManager.GetComponent<SUPAnimationBridge>();
        if (bridge != null)
        {
            so.FindProperty("animationBridge").objectReferenceValue = bridge;
        }

        so.ApplyModifiedProperties();
        EditorUtility.SetDirty(uiManager);
        Debug.Log("UIManager: tutti i riferimenti collegati automaticamente.");
    }

    // ------- Helper di layout -------

    private static void FixUIControl(GameObject control)
    {
        foreach (var t in control.GetComponentsInChildren<Text>(true))
        {
            t.verticalOverflow = VerticalWrapMode.Overflow;
            var rt = t.GetComponent<RectTransform>();
            if (Mathf.Approximately(rt.anchorMin.y, 0f) && Mathf.Approximately(rt.anchorMax.y, 1f))
            {
                rt.offsetMin = new Vector2(rt.offsetMin.x, 2);
                rt.offsetMax = new Vector2(rt.offsetMax.x, -2);
            }
        }
    }

    private static void AddSectionHeader(Transform parent, string text)
    {
        var t = AddPlainText(parent, text, 24);
        t.fontStyle = FontStyle.Bold;
        t.color = Color.white;
    }

    private static Text AddPlainText(Transform parent, string text, float height)
    {
        var go = DefaultControls.CreateText(Resources);
        go.transform.SetParent(parent, false);
        var t = go.GetComponent<Text>();
        t.text = text;
        t.color = new Color(0.92f, 0.92f, 0.92f);
        t.fontSize = 14;
        t.alignment = TextAnchor.UpperLeft;
        t.horizontalOverflow = HorizontalWrapMode.Wrap;
        t.verticalOverflow = VerticalWrapMode.Overflow;
        var le = go.AddComponent<LayoutElement>();
        le.preferredHeight = height;
        le.minHeight = height;
        return t;
    }

    private static GameObject AddFieldRow(Transform parent, string labelText, GameObject control, float rowHeight = 22)
    {
        var row = new GameObject("Row_" + labelText, typeof(RectTransform));
        row.transform.SetParent(parent, false);
        var hlg = row.AddComponent<HorizontalLayoutGroup>();
        hlg.childForceExpandWidth = false;
        hlg.childControlWidth = true;
        hlg.childControlHeight = true;
        hlg.childForceExpandHeight = true;
        hlg.spacing = 6;
        hlg.childAlignment = TextAnchor.MiddleLeft;
        var rowLE = row.AddComponent<LayoutElement>();
        rowLE.preferredHeight = rowHeight;
        rowLE.minHeight = rowHeight;

        var labelGO = DefaultControls.CreateText(Resources);
        labelGO.transform.SetParent(row.transform, false);
        var labelComp = labelGO.GetComponent<Text>();
        labelComp.text = labelText;
        labelComp.color = new Color(0.92f, 0.92f, 0.92f);
        labelComp.fontSize = 12;
        labelComp.alignment = TextAnchor.MiddleLeft;
        var labelLE = labelGO.AddComponent<LayoutElement>();
        labelLE.preferredWidth = 120;
        labelLE.minWidth = 120;

        control.transform.SetParent(row.transform, false);
        var fieldLE = control.AddComponent<LayoutElement>();
        fieldLE.flexibleWidth = 1;
        fieldLE.preferredHeight = rowHeight;

        return control;
    }

    private static Text AddSliderRow(Transform parent, string labelText, GameObject sliderGO, float rowHeight = 24)
    {
        var row = new GameObject("Row_" + labelText, typeof(RectTransform));
        row.transform.SetParent(parent, false);
        var hlg = row.AddComponent<HorizontalLayoutGroup>();
        hlg.childForceExpandWidth = false;
        hlg.childControlWidth = true;
        hlg.childControlHeight = true;
        hlg.childForceExpandHeight = true;
        hlg.spacing = 6;
        hlg.childAlignment = TextAnchor.MiddleLeft;
        var rowLE = row.AddComponent<LayoutElement>();
        rowLE.preferredHeight = rowHeight;
        rowLE.minHeight = rowHeight;

        var labelGO = DefaultControls.CreateText(Resources);
        labelGO.transform.SetParent(row.transform, false);
        var labelComp = labelGO.GetComponent<Text>();
        labelComp.text = labelText;
        labelComp.color = new Color(0.92f, 0.92f, 0.92f);
        labelComp.fontSize = 12;
        labelComp.alignment = TextAnchor.MiddleLeft;
        var labelLE = labelGO.AddComponent<LayoutElement>();
        labelLE.preferredWidth = 120;
        labelLE.minWidth = 120;

        sliderGO.transform.SetParent(row.transform, false);
        var sliderLE = sliderGO.AddComponent<LayoutElement>();
        sliderLE.flexibleWidth = 1;
        sliderLE.preferredHeight = rowHeight;

        var valueGO = DefaultControls.CreateText(Resources);
        valueGO.transform.SetParent(row.transform, false);
        var valueComp = valueGO.GetComponent<Text>();
        valueComp.text = "0.0";
        valueComp.color = new Color(0.92f, 0.92f, 0.92f);
        valueComp.fontSize = 12;
        valueComp.alignment = TextAnchor.MiddleRight;
        var valueLE = valueGO.AddComponent<LayoutElement>();
        valueLE.preferredWidth = 40;
        valueLE.minWidth = 40;

        return valueComp;
    }

    private static Button AddButton(Transform parent, string text, float height = 32, Color? btnColor = null, Color? textColor = null)
    {
        var go = DefaultControls.CreateButton(Resources);
        go.transform.SetParent(parent, false);
        var label = go.GetComponentInChildren<Text>();
        if (label != null)
        {
            label.text = text;
            if (textColor.HasValue) label.color = textColor.Value;
        }
        if (btnColor.HasValue)
        {
            go.GetComponent<Image>().color = btnColor.Value;
        }
        var le = go.AddComponent<LayoutElement>();
        le.preferredHeight = height;
        le.minHeight = height;
        return go.GetComponent<Button>();
    }
}
