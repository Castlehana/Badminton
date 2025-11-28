using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;

public class AchievementView : MonoBehaviour
{
    public TMP_Text winsText;
    public TMP_Text losesText;
    public TMP_Text streakText;

    public TMP_Text overText;
    public TMP_Text underText;


    void OnEnable()
    {
        var cur = SaveManager.Instance.Current;
        if (cur == null ) { winsText.text = losesText.text = streakText.text = "-"; return; }

        winsText.text = cur.achv.totalWins.ToString();
        losesText.text = cur.achv.totalLoses.ToString();
        streakText.text = cur.achv.highestStreak.ToString();
        overText.text = cur.achv.overSwing.ToString();
        underText.text = cur.achv.underSwing.ToString();
    }
}
