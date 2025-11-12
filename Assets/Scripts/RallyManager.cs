using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using UnityEngine;
using TMPro;

public enum RallyState
{
    Ready, Rallying, Checking, Ended
}
public enum ServeTurn
{
    MyTurn, AiTurn
}
public enum ModeState
{
    Easy, Normal, Hard, Training
}

public class RallyManager : MonoBehaviour
{
    public RallyState State;
    public ModeState Mode;
    public ServeTurn Turn;

    public int myScore = 0;
    public int aiScore = 0;

    public GameObject shuttlePrefab;
    Vector3 aiServePoint = new Vector3(0f, 3f, -7.5f);

    public TextMeshProUGUI playerText;
    public TextMeshProUGUI opponentText;

    private bool isResetting = false;
    private bool isAiServing = false;

    public MenuSceneLoader menuSceneLoader;

    // Start is called before the first frame update
    void Start()
    {
        State = RallyState.Ready;
        Turn = ServeTurn.MyTurn;

        myScore = 0;
        aiScore = 0;

        UnityEngine.Debug.Log($"Rally Start (my: {myScore}, ai: {aiScore})");
    }

    // Update is called once per frame
    void Update()
    {
        if (State == RallyState.Ended && !isResetting)
        {
            StartCoroutine(ReturnToReady());
        }
        if (State == RallyState.Ready && Turn == ServeTurn.AiTurn && !isAiServing)
        {
            StartCoroutine(AiServe());
        }
    }

    private IEnumerator AiServe()
    {
        isAiServing = true;

        yield return new WaitForSeconds(1.0f);

        UnityEngine.Debug.Log("Ai 서브!!");

        GameObject newShuttle = Instantiate(shuttlePrefab, aiServePoint, Quaternion.identity);
        Shuttlecock shuttle = newShuttle.GetComponent<Shuttlecock>();

        shuttle.Launch(0f, 45f, 20f);

        State = RallyState.Rallying;

        isAiServing = false ;
    }

    private IEnumerator ReturnToReady()
    {
        isResetting = true;

        yield return new WaitForSeconds(1.0f);

        // test here
        State = RallyState.Ready;

        isResetting = false;
    }

    public void PointCheck(bool mySide, bool opponentSide, bool inCourt, bool underNet)
    {
        // 득점 판정 처리
        if (underNet)
        {
            // 1. 네트 밑 통과 + 플레이어 코트 -> 플레이어 득점
            if (mySide)
            {
                myScore++;
                Turn = ServeTurn.MyTurn;
                UnityEngine.Debug.Log("Player Point");
            }
            // 2. 네트 밑 통과 + 상대 코트 -> 상대 득점
            if (opponentSide)
            {
                aiScore++;
                Turn = ServeTurn.AiTurn;
                UnityEngine.Debug.Log("AI Point");
            } 
        }
        else
        {
            if (mySide)
            {
                // 3. 플레이어 영역 + 인코트 -> 상대 득점
                if (inCourt)
                {
                    aiScore++;
                    Turn = ServeTurn.AiTurn;
                    UnityEngine.Debug.Log("AI Point");
                }
                // 4. 플레이어 영역 + 아웃코트 -> 플레이어 득점
                else
                {
                    myScore++;
                    Turn = ServeTurn.MyTurn;
                    UnityEngine.Debug.Log("Player Point");
                } 
            }
            else if (opponentSide)
            {
                // 5. 상대 영역 + 인코트 -> 플레이어 득점
                if (inCourt)
                {
                    myScore++;
                    Turn = ServeTurn.MyTurn;
                    UnityEngine.Debug.Log("Player Point");
                }
                // 6. 상대 영역 + 아웃코트 -> 상대 득점
                else
                {
                    aiScore++;
                    Turn = ServeTurn.AiTurn;
                    UnityEngine.Debug.Log("AI Point");
                }
            }
        }
        UpdateScoreUI(myScore, aiScore);
        ScoreCheck();
    }
    public void ScoreCheck()
    {
        if (Mathf.Abs(myScore - aiScore) >= 2 && myScore >= 21)
        {
            UnityEngine.Debug.Log("You Win!");
            menuSceneLoader.LoadWinScene();
            return;
        }
        else if (Mathf.Abs(myScore - aiScore) >= 2 && aiScore >= 21)
        {
            UnityEngine.Debug.Log("You Lose!");
            menuSceneLoader.LoadLoseScene();
            return;
        }

        ResetPosition();


        State = RallyState.Ended;
    }

    public void UpdateScoreUI(int player, int opponent)
    {
        if (playerText)
            playerText.text = "Player : " + player.ToString();
        if (opponentText)
            opponentText.text = "Opponent : " + opponent.ToString();
    }

    public void ResetPosition()
    {
        //플레이어와 Ai의 위치 리셋
    }
}
