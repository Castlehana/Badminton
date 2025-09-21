using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class RallyJudge : MonoBehaviour
{
    [Header("참조")]
    public ScoreManager score;
    public string shuttleTag = "Shuttlecock";

    [Header("코트 설정")]
    public float netZ = 0f;         // 네트 z 위치
    public float groundY = 0f;      // 바닥 y 높이
    public float halfWidthX = 11f;  // 코트 반폭
    public float halfLengthZ = 20f; // 코트 반장

    [Header("디버그")]
    public bool debugLogs = false;
    public bool drawGizmos = true;

    private Shuttlecock active;
    private Vector3 prevPos;
    private bool judgedThisShuttle = false;

    // 언더넷 플래그 (UnderNetZone에서 MarkUnderNet() 호출로 세팅)
    private bool underNet = false;

    // 착지 임계값 = groundY + 셔틀 콜라이더 bounds.extents.y
    private float groundContactY = 0f;

    // Start is called before the first frame update
    void Start()
    {
        FindActiveShuttle();
    }

    // Update is called once per frame
    void Update()
    {
        if (active == null)
        {
            FindActiveShuttle();
            return;
        }

        Vector3 now = active.transform.position;

        // 착지 순간 감지: 위에서 아래로 groundY 통과
        if (!judgedThisShuttle && prevPos.y > groundY && now.y <= groundY)
        {
            judgedThisShuttle = true;

            // 착지한 쪽
            bool playerSide = now.z < netZ;
            var landingSide = playerSide ? ScoreManager.Side.Player : ScoreManager.Side.Opponent;

            // 인코트 여부
            bool inBounds = Mathf.Abs(now.x) <= halfWidthX && Mathf.Abs(now.z) <= halfLengthZ;

            ScoreManager.Side pointTo;

            if (!underNet)
            {
                // 정상 규칙
                pointTo = inBounds
                    ? OpponentOf(landingSide) // 인 → 반대편 득점
                    : landingSide;            // 아웃 → 그쪽 득점
            }
            else
            {
                // 언더넷 → 무조건 떨어진 쪽 득점 (친 쪽 실수)
                pointTo = landingSide;
            }

            if (debugLogs) Debug.Log($"[RallyJudge] Landed. underNet={underNet}, inBounds={inBounds}, landingSide={landingSide} → pointTo={pointTo}");
            score.AwardPoint(pointTo);
        }

        prevPos = now;

        // 셔틀이 파괴되면 다음 랠리 준비
        if (active == null || active.gameObject == null)
        {
            active = null;
            judgedThisShuttle = false;
            underNet = false;
        }
    }

    /// <summary>UnderNet 트리거에서 호출: 언더넷 플래그 세팅</summary>
    public void MarkUnderNet() { underNet = true; if (debugLogs) Debug.Log("[RallyJudge] UnderNet = TRUE"); }


    private ScoreManager.Side OpponentOf(ScoreManager.Side s)
        => s == ScoreManager.Side.Player ? ScoreManager.Side.Opponent : ScoreManager.Side.Player;

    private void FindActiveShuttle()
    {
        var go = GameObject.FindGameObjectWithTag(shuttleTag);
        if (go == null)
        {
            active = null; judgedThisShuttle = false; underNet = false;
            return;
        }

        active = go.GetComponent<Shuttlecock>();
        judgedThisShuttle = false; underNet = false;

        if (active != null)
        {
            prevPos = active.transform.position;

            // 콜라이더 반지름/반높이만큼 보정해서 '중심이 바닥에 닿는 y'를 임계값으로 사용
            var col = active.GetComponent<Collider>();
            float halfHeight = col ? col.bounds.extents.y : 0.5f; // SphereCollider(0.5) 기본값
            groundContactY = groundY + halfHeight;

            if (debugLogs) Debug.Log($"[RallyJudge] New shuttle. groundContactY={groundContactY:F3} (ground {groundY} + halfHeight {halfHeight})");
        }
    }

    void OnDrawGizmosSelected()
    {
        if (!drawGizmos) return;
        // 네트 평면
        Gizmos.color = Color.red;
        Gizmos.DrawLine(new Vector3(-halfWidthX, 0, netZ), new Vector3(halfWidthX, 0, netZ));
        // 착지 임계 y (대략 표시)
        Gizmos.color = new Color(0, 1, 0, 0.35f);
        float y = Application.isPlaying ? groundContactY : groundY + 0.5f;
        Gizmos.DrawCube(new Vector3(0, y, 0), new Vector3(halfWidthX * 2f, 0.01f, halfLengthZ * 2f));
    }
}
