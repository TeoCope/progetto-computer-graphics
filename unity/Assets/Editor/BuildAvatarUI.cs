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
        vlg.childControlHeight = false;
        vlg.spacing = 6;
        vlg.padding = new RectOffset(8, 8, 8, 8);
        content.gameObject.AddComponent<ContentSizeFitter>().verticalFit = ContentSizeFitter.FitMode.PreferredSize;

        // --- Preset e genere ---
        AddSectionHeader(content, "Preset");
        var presetDropdown = AddFieldRow(content, "Preset", DefaultControls.CreateDropdown(Resources)).GetComponent<Dropdown>();
        var presetDescriptionText = AddPlainText(content, "", 40);
        var genderDropdown = AddFieldRow(content, "Genere", DefaultControls.CreateDropdown(Resources)).GetComponent<Dropdown>();

        // --- Misure target ---
        AddSectionHeader(content, "Misure target (cm)");
        var measurementInputFields = new List<InputField>();
        foreach (var label in MeasurementLabels)
        {
            var fieldGO = DefaultControls.CreateInputField(Resources);
            var field = fieldGO.GetComponent<InputField>();
            field.contentType = InputField.ContentType.DecimalNumber;
            AddFieldRow(content, label, fieldGO);
            measurementInputFields.Add(field);
        }
        var fitButton = AddButton(content, "Fit to Measurements");

        // --- Misure attuali ---
        AddSectionHeader(content, "Misure attuali");
        var measurementsText = AddPlainText(content, "", 160);

        // --- Betas (avanzato) ---
        AddSectionHeader(content, "Controlli avanzati (betas)");
        var betaSliders = new List<Slider>();
        var betaValueLabels = new List<Text>();
        for (int i = 0; i < 10; i++)
        {
            var sliderGO = DefaultControls.CreateSlider(Resources);
            var slider = sliderGO.GetComponent<Slider>();
            slider.minValue = -5f;
            slider.maxValue = 5f;
            slider.value = 0f;
            var valueLabel = AddSliderRow(content, $"Beta {i}", sliderGO);
            betaSliders.Add(slider);
            betaValueLabels.Add(valueLabel);
        }
        var resetButton = AddButton(content, "Reset");

        // --- Carica JSON ---
        AddSectionHeader(content, "Carica JSON esportato");
        var jsonInputGO = DefaultControls.CreateInputField(Resources);
        var jsonPathInput = jsonInputGO.GetComponent<InputField>();
        AddFieldRow(content, "Percorso", jsonInputGO);
        var loadJsonButton = AddButton(content, "Load JSON");

        // --- Stato ---
        AddSectionHeader(content, "Stato");
        var statusText = AddPlainText(content, "", 60);

        WireUIManager(presetDropdown, presetDescriptionText, genderDropdown,
            measurementInputFields, fitButton, measurementsText,
            betaSliders, betaValueLabels, resetButton, jsonPathInput, loadJsonButton, statusText);

        Selection.activeGameObject = canvasGO;
        EditorUtility.DisplayDialog("Avatar UI",
            "Canvas creata. Se in scena esiste già un GameObject con UIManager i riferimenti sono " +
            "stati collegati automaticamente; altrimenti crea prima 'AvatarManager' (punti 6-7 del " +
            "README) e rilancia questo comando per il collegamento automatico.", "OK");
    }

    private static void WireUIManager(Dropdown presetDropdown, Text presetDescriptionText, Dropdown genderDropdown,
        List<InputField> measurementInputFields, Button fitButton, Text measurementsText,
        List<Slider> betaSliders, List<Text> betaValueLabels, Button resetButton,
        InputField jsonPathInput, Button loadJsonButton, Text statusText)
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

        so.FindProperty("resetButton").objectReferenceValue = resetButton;
        so.FindProperty("jsonPathInput").objectReferenceValue = jsonPathInput;
        so.FindProperty("loadJsonButton").objectReferenceValue = loadJsonButton;
        so.FindProperty("statusText").objectReferenceValue = statusText;

        so.ApplyModifiedProperties();
        EditorUtility.SetDirty(uiManager);
        Debug.Log("UIManager: tutti i riferimenti collegati automaticamente.");
    }

    // ------- Helper di layout -------

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

    private static GameObject AddFieldRow(Transform parent, string labelText, GameObject control, float rowHeight = 26)
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
        labelLE.preferredWidth = 60;
        labelLE.minWidth = 60;

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

    private static Button AddButton(Transform parent, string text, float height = 32)
    {
        var go = DefaultControls.CreateButton(Resources);
        go.transform.SetParent(parent, false);
        var label = go.GetComponentInChildren<Text>();
        if (label != null) label.text = text;
        var le = go.AddComponent<LayoutElement>();
        le.preferredHeight = height;
        le.minHeight = height;
        return go.GetComponent<Button>();
    }
}
