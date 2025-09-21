using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using UnityEngine;
using UnityEngine.Events;

/// <summary>
/// 한 "게임"의 점수와 종료 조건을 관리하는 스크립트.
/// - 목표 점수(기본 11)에 먼저 도달한 쪽이 이기면 게임 종료
/// - 점수 변화/포인트 득점/게임 종료를 UnityEvent로 외부에 알림
/// - 외부(예: RallyJudge)가 '누가 점수를 땄는지'만 알려주면 나머지(합산/종료판정)는 여기서 처리
/// </summary>

public class ScoreManager : MonoBehaviour
{
    public enum Side { Player, Opponent }

    [Header("라운드 점수")]
    public int gamePoint = 11;

    [Header("점수")]
    public int player = 0;
    public int opponent = 0;

    [Header("이벤트")]
    public UnityEvent<int, int> onScoreChanged;
    public UnityEvent<Side> onPointWon;
    public UnityEvent<Side> onGameOver;

    // 11점 내기 게임
    public bool IsGameOver => (player >= gamePoint ||  opponent >= gamePoint) /*&& Mathf.Abs(player - opponent) >= 2*/;

    public void ResetAll() { player = 0; opponent = 0; onScoreChanged?.Invoke(player, opponent); }

    public void AwardPoint(Side who)
    {
        if (IsGameOver) return;

        if (who == Side.Player)
            player++;
        else
            opponent++;

        onScoreChanged?.Invoke(player, opponent);
        onPointWon?.Invoke(who);

        if (IsGameOver) onGameOver?.Invoke(who);
    }

    // Start is called before the first frame update
    void Start()
    {
        // 시작 시 라운드 리셋
        ResetAll();
    }

    // Update is called once per frame
    void Update()
    {
        if (Input.GetKeyDown(KeyCode.N))
        {
            AwardPoint(ScoreManager.Side.Player);
        }

        if (Input.GetKeyDown(KeyCode.M))
        {
            AwardPoint(ScoreManager.Side.Opponent);
        }
    }
}
