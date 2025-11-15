using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;

public class ScoreUI : MonoBehaviour
{
    public TextMeshProUGUI playerText;
    public TextMeshProUGUI opponentText;

    public void UpdateScore(int player, int opponent)
    {
        if (playerText)
            playerText.text = "Player : " + player.ToString();
        if (opponentText)
            opponentText.text = "Opponent : " + opponent.ToString();
    }

    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
