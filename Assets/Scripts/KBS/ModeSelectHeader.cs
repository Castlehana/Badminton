using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;

public class ModeSelectHeader : MonoBehaviour
{
    [Header("표시 위치 (Text 또는 TMP_Text 중 하나만 연결)")]
    public TMP_Text tmpText;

    [Header("선택 안 했을때 문구")]
    [SerializeField] string fallback = "No Slot Selected";

    // Start is called before the first frame update
    void Start()
    {
        Refresh();
    }

    public void Refresh()
    {
        var sm = SaveManager.Instance;
        string msg;

        if (sm != null && sm.Current != null)
        {
            var d = sm.Current;
            msg = $"Slot {d.slotIndex}, {d.profileName}";
        }
        else
        {
            msg = fallback;
        }

        if (tmpText) tmpText.text = msg;
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
