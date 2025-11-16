using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;
using TMPro;

public class SaveSlotButton : MonoBehaviour
{
    public int slotIndex = 1;   // 1, 2, 3
    public TMP_Text titleText;      // empty slot or summary
    public TMP_InputField nameInput;

    void OnEnable()
    {
        if (titleText != null)
            titleText.text = SaveManager.GetSummaryText(slotIndex);
    }

    // 슬롯 버튼 클릭: 있으면 로드-> 다음 씬, 없으면 생성-> 다음 씬
    public void OnClickSelect()
    {
        var mgr = SaveManager.Instance;

        if (SaveManager.Exists(slotIndex))
        {
            UnityEngine.Debug.Log("이미 있네");
            bool ok = mgr.Load(slotIndex);
            if (!ok) { titleText.text = "Corrupted Slot"; return; }
        }
        else
        {
            UnityEngine.Debug.Log("아직 없네");
            string pname = nameInput != null && !string.IsNullOrWhiteSpace(nameInput.text) ? nameInput.text : $"Player{slotIndex}";
            mgr.CreateNew(slotIndex, pname);
        }

        // 슬롯을 잡은 뒤 모드 선택 씬으로 이동
        SceneManager.LoadScene("SelectModeMenu");
    }

    public void OnClickDelete()
    {
        SaveManager.Delete(slotIndex);
        if (titleText != null) titleText.text = "Empty Slot";
    }
}
