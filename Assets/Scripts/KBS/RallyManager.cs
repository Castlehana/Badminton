using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using UnityEngine;
using TMPro;
using UnityEngine.UI;   // ← 추가

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

    public int gamePoint = 5;

    public GameObject shuttlePrefab;
    Vector3 aiServePoint = new Vector3(0f, 3f, -7.5f);

    public TextMeshProUGUI playerText;
    public TextMeshProUGUI opponentText;

    public TextMeshProUGUI AODplayerText;
    public TextMeshProUGUI AODopponentText;

    private bool isResetting = false;
    private bool isAiServing = false;

    [Header("점수 UI")]
    public RectTransform scorePanel;              // 점수판 패널
    public TextMeshProUGUI serveText;            // 서브!
    public float scorePanelMoveDuration = 0.4f;  // 점수판 올라오는 시간
    public float scoreHoldDuration = 1.2f;       // 점수 바뀌고 얼마 기다릴지
    public float serveTextDuration = 2.5f;       // 서브! 글씨 띄우는 시간

    Vector2 _scorePanelShownPos;
    Vector2 _scorePanelHiddenPos;
    bool _pointSequenceRunning = false;

    public MenuSceneLoader menuSceneLoader;

    // 인스펙터에서 Enemy 오브젝트를 드래그해서 넣기
    public GameObject enemyObject;


    [Header("Ready 페이드용 이미지")]
    public Image readyImage;          // 비활성화 되어 있는 Image
    public float readyFadeDuration = 0.5f;
    public ScreenFreezeEffect screenFreeze;  // 인스펙터에서 할당



    // Start is called before the first frame update
    void Start()
    {
        State = RallyState.Ready;
        Turn = ServeTurn.MyTurn;

        myScore = 0;
        aiScore = 0;

        UnityEngine.Debug.Log($"Rally Start (my: {myScore}, ai: {aiScore})");

        // 점수판 위치 기록
        if (scorePanel != null)
        {
            _scorePanelShownPos = scorePanel.anchoredPosition;
            _scorePanelHiddenPos = _scorePanelShownPos + new Vector2(0f, -200f);

            // 처음에는 숨겨두기
            scorePanel.gameObject.SetActive(false);
        }

        if (serveText != null)
        {
            serveText.gameObject.SetActive(false);
        }

        // Fade 전환
        if (readyImage != null)
        {
            // 처음엔 꺼져 있고, 알파 0으로
            Color c = readyImage.color;
            c.a = 0f;
            readyImage.color = c;
            readyImage.gameObject.SetActive(false);
        }


    }

    // Update is called once per frame
    void Update()
    {
        // ★ Ready 상태일 때 Enemy 위치 고정 (x=0, z=-10)
        if (State == RallyState.Ready && enemyObject != null)
        {
            Vector3 pos = enemyObject.transform.position;
            pos.x = 0f;
            pos.z = -10f;
            enemyObject.transform.position = pos;
        }
        // Ready가 아닐 때는 위치를 건드리지 않으므로 "고정 풀림"
        /*
        if (State == RallyState.Ended && !isResetting)
        {
            StartCoroutine(ReturnToReady());
        }
        if (State == RallyState.Ready && Turn == ServeTurn.AiTurn && !isAiServing)
        {
            StartCoroutine(AiServe());
        }*/
    }

    private IEnumerator AiServe()
    {
        isAiServing = true;

        yield return new WaitForSeconds(1.0f);

        UnityEngine.Debug.Log("Ai 서브!!");

        GameObject newShuttle = Instantiate(shuttlePrefab, aiServePoint, Quaternion.identity);
        Shuttlecock shuttle = newShuttle.GetComponent<Shuttlecock>();

        shuttle.Launch(0f, 45f, 15f);

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
        int prevMyScore = myScore;
        int prevAiScore = aiScore;

        if (Mode == ModeState.Training)
        {
            StartCoroutine(ReturnToReady());
            return;
        }


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
        bool playerWonPoint = myScore > prevMyScore;

        // 점수 연출 코루틴 시작
        if (!_pointSequenceRunning)
        {
            StartCoroutine(PointSequence(prevMyScore, prevAiScore, playerWonPoint));
        }
        else
        {
            UpdateScoreUI(myScore, aiScore);
            ScoreCheck();
        }
    }

    public bool ScoreCheck()
    {
        if (Mathf.Abs(myScore - aiScore) >= 2 && myScore >= gamePoint)
        {
            // achv 수정
            var mgr = SaveManager.Instance;
            if (mgr != null && mgr.Current != null)
            {
                mgr.Current.achv.totalWins++;
                // 연승 계산
                mgr.Current.achv.streak++;
                mgr.Current.achv.highestStreak = Mathf.Max(mgr.Current.achv.highestStreak, mgr.Current.achv.streak);
                mgr.Save();
            }

            UnityEngine.Debug.Log("You Win!");
            menuSceneLoader.LoadWinScene();
            return true;
        }
        else if (Mathf.Abs(myScore - aiScore) >= 2 && aiScore >= gamePoint)
        {
            // achv 수정
            var mgr = SaveManager.Instance;
            if (mgr != null && mgr.Current != null)
            {
                mgr.Current.achv.totalLoses++;
                mgr.Current.achv.streak = 0;
                mgr.Save();
            }

            UnityEngine.Debug.Log("You Lose!");
            menuSceneLoader.LoadLoseScene();
            return true;
        }

        return false;
    }

    public void UpdateScoreUI(int player, int opponent)
    {
        if (playerText)
            playerText.text =  player.ToString();
        if (AODplayerText)
            AODplayerText.text = player.ToString();
        if (opponentText)
            opponentText.text = opponent.ToString();
        if (AODopponentText)
            AODopponentText.text = opponent.ToString();
    }
    public void ResetPosition()
    {
        // 인스펙터에서 할당한 Enemy 오브젝트의 X/Z만 리셋
        if (enemyObject != null)
        {
            Vector3 pos = enemyObject.transform.position;
            pos.x = 0f;
            pos.z = -10f;
            enemyObject.transform.position = pos;
        }
        else
        {
            UnityEngine.Debug.LogWarning("RallyManager: enemyObject가 할당되지 않았습니다.");
        }
    }

    // 추가 
    // +++ 점수 연출 코드
    private IEnumerator PointSequence(int prevMyScore, int prevAiScore, bool playerWonPoint)
    {
        _pointSequenceRunning = true;

        // 1) 애들 움직임 정지
        float prevTimeScale = Time.timeScale;
        Time.timeScale = 0f;

        // 2) 점수판 효과
        if (scorePanel != null)
        {
            // 점수판 활성화
            scorePanel.gameObject.SetActive(true);
            // 이전 점수 상태로 초기화
            UpdateScoreUI(prevMyScore, prevAiScore);

            // 아래에서 위로 슬라이드 인
            Vector2 from = _scorePanelHiddenPos;
            Vector2 to = _scorePanelShownPos;
            float t = 0f;
            while (t < scorePanelMoveDuration)
            {
                t += Time.unscaledDeltaTime;
                float alpha = Mathf.Clamp01(t / scorePanelMoveDuration);
                float eased = alpha * alpha * (3f - 2f * alpha);
                scorePanel.anchoredPosition = Vector2.Lerp(from, to, eased);
                yield return null;
            }
            scorePanel.anchoredPosition = to;

            // 점수 득점한 거로 바꾸기
            yield return new WaitForSecondsRealtime(0.3f);
            UpdateScoreUI(myScore, aiScore);

            // 기다리기..
            yield return new WaitForSecondsRealtime(scoreHoldDuration);

            // 점수판 숨기기
            scorePanel.gameObject.SetActive(false);
        }
        else
        {
            // 예외
            UpdateScoreUI(myScore, aiScore);
            yield return new WaitForSecondsRealtime(1.0f);
        }

        // 3) 서브! 텍스트 표시
        if (serveText != null)
        {
            if (Mathf.Abs(myScore - aiScore) >= 2 && ((myScore >= gamePoint) || (aiScore >= gamePoint)))
            { }
            else
            {
                serveText.gameObject.SetActive(true);
                serveText.text = "Start!";
                yield return new WaitForSecondsRealtime(serveTextDuration);
                serveText.gameObject.SetActive(false);
            }
        }

        screenFreeze.PlayFreeze();
        yield return new WaitForSecondsRealtime(0.5f);

        // 4) 애들 움직임 재개
        Time.timeScale = prevTimeScale;

        // 5) 게임 종료 여부 확인
        bool gameOver = ScoreCheck();

        // 6) 게임이 끝나지 않았다면 다음 랠리 준비 및 서브
        if (!gameOver)
        {
            ResetPosition();
            State = RallyState.Ready;
            // ② 0.5초 기다리기
            //yield return new WaitForSecondsRealtime(0.5f);

            // ③ Ready가 된 "후"에 FadeOut
            if (readyImage != null)
            {
                //yield return StartCoroutine(FadeOutReadyImage());
            }



            if (Turn == ServeTurn.AiTurn && !isAiServing)
            {
                StartCoroutine(AiServe());
            }
        }

        _pointSequenceRunning = false;
    }

    // ==================== Ready 이미지 페이드 인/아웃 ====================

    private IEnumerator FadeInReadyImage()
    {
        if (readyImage == null) yield break;

        // readyImage 포함, 모든 자식 Image / TextMeshProUGUI 가져오기
        Image[] images = readyImage.GetComponentsInChildren<Image>(true);
        TextMeshProUGUI[] texts = readyImage.GetComponentsInChildren<TextMeshProUGUI>(true);

        readyImage.gameObject.SetActive(true);

        // 시작은 알파 0
        foreach (var img in images)
        {
            Color c = img.color;
            c.a = 0f;
            img.color = c;
        }
        foreach (var txt in texts)
        {
            Color c = txt.color;
            c.a = 0f;
            txt.color = c;
        }

        float t = 0f;
        while (t < readyFadeDuration)
        {
            t += Time.unscaledDeltaTime;  // timeScale=0이어도 동작
            float alpha = Mathf.Clamp01(t / readyFadeDuration);

            foreach (var img in images)
            {
                Color c = img.color;
                c.a = alpha;
                img.color = c;
            }
            foreach (var txt in texts)
            {
                Color c = txt.color;
                c.a = alpha;
                txt.color = c;
            }

            yield return null;
        }

        // 마지막 보정
        foreach (var img in images)
        {
            Color c = img.color;
            c.a = 1f;
            img.color = c;
        }
        foreach (var txt in texts)
        {
            Color c = txt.color;
            c.a = 1f;
            txt.color = c;
        }
    }
    private IEnumerator FadeOutReadyImage()
    {
        if (readyImage == null) yield break;

        Image[] images = readyImage.GetComponentsInChildren<Image>(true);
        TextMeshProUGUI[] texts = readyImage.GetComponentsInChildren<TextMeshProUGUI>(true);

        // 시작은 알파 1
        foreach (var img in images)
        {
            Color c = img.color;
            c.a = 1f;
            img.color = c;
        }
        foreach (var txt in texts)
        {
            Color c = txt.color;
            c.a = 1f;
            txt.color = c;
        }

        float t = 0f;
        while (t < readyFadeDuration)
        {
            t += Time.unscaledDeltaTime;
            float alpha = 1f - Mathf.Clamp01(t / readyFadeDuration);

            foreach (var img in images)
            {
                Color c = img.color;
                c.a = alpha;
                img.color = c;
            }
            foreach (var txt in texts)
            {
                Color c = txt.color;
                c.a = alpha;
                txt.color = c;
            }

            yield return null;
        }

        foreach (var img in images)
        {
            Color c = img.color;
            c.a = 0f;
            img.color = c;
        }
        foreach (var txt in texts)
        {
            Color c = txt.color;
            c.a = 0f;
            txt.color = c;
        }

        readyImage.gameObject.SetActive(false);
    }



}

