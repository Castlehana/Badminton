using System.Linq;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

[RequireComponent(typeof(PlayerMovement), typeof(Rigidbody))]
public class PlayerAgent : Agent
{
    [Header("Refs")]
    public PlayerMovement movement;
    public EnemyShooting shooting;                // 스윙 실행용(임시 사용)
    public ReinforcementLearningManager rl;       // courtHalfWidthX/Z 등 값 참조

    [Header("Tags/Names")]
    public string shuttlecockTag = "Shuttlecock"; // 인스펙터에서 "셔틀콕"으로 바꿔도 됨
    public string goalTag = "Goal";               // 인스펙터에서 "골"로 바꿔도 됨

    [Header("Rewards")]
    public float stepCenteringWeight = 0.01f;     // 골 지점 근접 shaping (프레임당, 내 코트일 때만)
    public float hitReward = 0.5f;                // 스윙+타격 성공 즉시 보상(소)
    public float landReward = 2.0f;               // 셔틀이 상대 코트 인존에 "들어오는 순간" 보상(대)
    public float timePenalty = -0.0005f;          // 소량 지연 패널티
    public float invalidSwingPenalty = -0.001f;   // 헛스윙 소패널티

    [Header("Episode/Rally")]
    public float postLandGrace = 0.0f;            // 착지 후 종료 지연(미사용시 0)
    GameObject _trackedShuttle;                   // 이번 랠리 추적 셔틀(에피소드 종료 판단용)

    [Header("Court Areas")]
    public Collider myCourtArea;                  // 내 코트(Trigger 권장) — 골 근접 보상 판단용
    public Collider opponentInZone;               // 상대 코트 인존(Trigger 권장)

    [Header("Hit Zone")]
    public Collider hitZone;                      // "칠 수 있는 존" (Trigger 권장, 플레이어 앞/라켓 위치에 배치)

    Rigidbody _rb;

    // 보상/판정 플래그
    bool _awaitingLandReward;                     // 스윙 후 득점 판정 대기중?
    bool _wasInsideOpp;                           // 직전 프레임: 셔틀이 상대 인존 내부였나?

    public override void Initialize()
    {
        if (!movement) movement = GetComponent<PlayerMovement>();
        if (!_rb) _rb = GetComponent<Rigidbody>();
        if (!rl) rl = FindObjectOfType<ReinforcementLearningManager>();
        if (movement) movement.trainingMode = true; // 학습 모드: 외부 입력만
    }

    public override void OnEpisodeBegin()
    {
        _rb.velocity = Vector3.zero;
        movement.SetMoveInput(Vector2.zero);

        _trackedShuttle = null;

        _awaitingLandReward = false;
        _wasInsideOpp = false;
    }

    // ---------- 관측 ----------
    // 총 11 floats: (pX,pZ) + (gX,gZ,gOut) + (rel x,y,z) + (sc xNorm,zNorm,speedNorm)
    public override void CollectObservations(VectorSensor sensor)
    {
        float cw = rl ? rl.courtHalfWidthX : 11f;
        float cz = rl ? rl.courtHalfLengthZ : 20f;

        // 1) Player 정규화 위치
        Vector3 p = transform.position;
        sensor.AddObservation(Mathf.Clamp(p.x, -cw, cw) / cw);
        sensor.AddObservation(Mathf.Clamp(p.z, -cz, cz) / cz);

        // 2) Goal (예상 낙하지점)
        var goal = GameObject.FindGameObjectWithTag(goalTag);
        float gX = 0f, gZ = 0f, gOut = 0f;
        if (goal)
        {
            Vector3 g = goal.transform.position;
            gX = Mathf.Clamp(g.x, -cw, cw) / cw;
            gZ = Mathf.Clamp(g.z, g.z >= 0 ? 0 : -cz, g.z >= 0 ? cz : 0) / cz;

            float gxRaw = g.x / cw, gzRaw = g.z / cz;
            gOut = (Mathf.Abs(gxRaw) > 1f || Mathf.Abs(gzRaw) > 1f) ? 1f : 0f;
        }
        sensor.AddObservation(gX); sensor.AddObservation(gZ); sensor.AddObservation(gOut);

        // 3) 셔틀 상태(가장 가까운 1개)
        var sc = FindNearestShuttlecock();
        Vector3 rel = Vector3.zero;
        float scxN = 0f, sczN = 0f, speedN = 0f;
        if (sc)
        {
            rel = sc.transform.position - transform.position;

            Vector3 sp = sc.transform.position;
            scxN = Mathf.Clamp(sp.x, -cw, cw) / cw;
            sczN = Mathf.Clamp(sp.z, sp.z >= 0 ? 0 : -cz, sp.z >= 0 ? cz : 0) / cz;

            var scRb = sc.GetComponent<Rigidbody>();
            float spd = scRb ? scRb.velocity.magnitude : 0f;
            speedN = Mathf.Clamp(spd / 50f, 0f, 1f);
        }

        sensor.AddObservation(Mathf.Clamp(rel.x / cw, -1f, 1f));
        sensor.AddObservation(Mathf.Clamp(rel.y / 10f, -1f, 1f));
        sensor.AddObservation(Mathf.Clamp(rel.z / cz, -1f, 1f));
        sensor.AddObservation(scxN);
        sensor.AddObservation(sczN);
        sensor.AddObservation(speedN);
    }

    // ---------- 통계 ----------
    void Stat(string key, float val)
    {
        Academy.Instance.StatsRecorder.Add(key, val, StatAggregationMethod.Average);
    }

    // ---------- 유틸 ----------
    bool IsInside(Collider col, Vector3 point)
    {
        if (!col || col.Equals(null)) return false;
        // ClosestPoint가 point와 같다면 내부(Trigger 포함)로 간주
        return (col.ClosestPoint(point) - point).sqrMagnitude < 1e-6f;
    }

    GameObject FindNearestShuttlecock()
    {
        var all = GameObject.FindGameObjectsWithTag(shuttlecockTag);
        if (all == null || all.Length == 0) return null;
        var me = transform.position;
        GameObject best = null; float bestD = float.MaxValue;
        foreach (var go in all)
        {
            float d = (go.transform.position - me).sqrMagnitude;
            if (d < bestD) { bestD = d; best = go; }
        }
        return best;
    }

    // ---------- 행동/보상 ----------
    // 연속 2 (move XZ) + 이산 4 (오버2, 언더2)
    public override void OnActionReceived(ActionBuffers actions)
    {
        var ca = actions.ContinuousActions;
        var da = actions.DiscreteActions;

        // 1) 이동
        Vector2 move = new Vector2(
            Mathf.Clamp(ca[0], -1f, 1f),
            Mathf.Clamp(ca[1], -1f, 1f)
        );
        movement.SetMoveInput(move);

        // 2) 스윙: 헛스윙도 실행(학습 신호로 사용)
        int swing = da[0]; // 0=없음, 1=OverStrong, 2=OverWeak, 3=UnderStrong, 4=UnderWeak
        if (swing != 0)
        {
            // (A) 애니메/액션은 항상 발생
            if (shooting)
            {
                if (swing == 1) shooting.OverStrong();
                else if (swing == 2) shooting.OverWeak();
                else if (swing == 3) shooting.UnderStrong();
                else if (swing == 4) shooting.UnderWeak();
            }

            // (B) 보상/패널티는 "hitZone 안에 셔틀이 있는가?"로 판정
            bool canHit = false;
            var sc = FindNearestShuttlecock();
            if (sc && hitZone)
            {
                canHit = IsInside(hitZone, sc.transform.position);
            }

            if (canHit)
            {
                _awaitingLandReward = true;       // 이후 상대 인존 "닿는 순간" 체크
                AddReward(hitReward);             // 타격 성공 즉시 보상(작게)
                Stat("player/hit_reward", hitReward);
            }
            else
            {
                AddReward(invalidSwingPenalty);   // 헛스윙 소패널티
                Stat("player/invalid_swing", 1f);
            }
        }

        // 3) 시간 패널티 + 골 지점 근접 shaping(내 코트일 때만)
        float stepReward = timePenalty;

        var goal = GameObject.FindGameObjectWithTag(goalTag);
        if (goal && rl)
        {
            bool goalInMyCourt = myCourtArea ? IsInside(myCourtArea, goal.transform.position) : false;
            if (goalInMyCourt)
            {
                float cw = rl.courtHalfWidthX, cz = rl.courtHalfLengthZ;
                Vector2 pN = new Vector2(
                    Mathf.Clamp(transform.position.x, -cw, cw) / cw,
                    Mathf.Clamp(transform.position.z, -cz, cz) / cz
                );
                Vector2 gN = new Vector2(
                    Mathf.Clamp(goal.transform.position.x, -cw, cw) / cw,
                    Mathf.Clamp(goal.transform.position.z, -cz, cz) / cz
                );

                // 0(멀다) ~ 1(가깝다) 근접도
                float dist = Vector2.Distance(pN, gN);
                float close = Mathf.Clamp01(1f - (dist / 1.4142f)); // sqrt(2)
                stepReward += stepCenteringWeight * close;
                Stat("player/goal_closeness", close);
            }
        }

        AddReward(stepReward);
        Stat("player/step_reward", stepReward);
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var ca = actionsOut.ContinuousActions;
        var da = actionsOut.DiscreteActions;

        ca[0] = Input.GetAxis("Horizontal");
        ca[1] = Input.GetAxis("Vertical");

        int swing = 0;
        if (Input.GetKey(KeyCode.Space)) swing = 1;  // OverStrong
        else if (Input.GetKey(KeyCode.Q)) swing = 3; // UnderStrong
        da[0] = swing;
    }

    void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.CompareTag(shuttlecockTag))
            Stat("player/contact_count", 1f); // 충돌 통계만
    }

    // ---------- 프레임 판정: 인존 "닿는 순간" 감지 & 에피소드 관리 ----------
    void LateUpdate()
    {
        // 이번 랠리의 셔틀 추적(없으면 가장 가까운 셔틀 하나 고정)
        if (_trackedShuttle == null)
            _trackedShuttle = FindNearestShuttlecock();

        var sc = _trackedShuttle;
        if (sc)
        {
            Vector3 sp = sc.transform.position;

            // 상대 인존 내부 여부(현재 프레임)
            bool insideOpp = opponentInZone ? IsInside(opponentInZone, sp) : false;

            // "들어온 순간(엣지)"에만 보상
            if (_awaitingLandReward)
            {
                if (insideOpp && !_wasInsideOpp)
                {
                    AddReward(landReward);
                    Stat("player/land_reward", landReward);
                    _awaitingLandReward = false;
                }
            }

            // 내부 상태 저장(엣지 검출용)
            _wasInsideOpp = insideOpp;
        }

        // 셔틀이 제거되면(스폰 시스템이 Destroy) 에피소드 종료
        bool trackedGone = _trackedShuttle == null || _trackedShuttle.Equals(null);
        if (!trackedGone) trackedGone = (_trackedShuttle == null);
        if (trackedGone)
        {
            Stat("player/episode_reward", GetCumulativeReward());
            Stat("player/episode_length", StepCount);

            EndEpisode();

            _trackedShuttle = null;
            _awaitingLandReward = false;
            _wasInsideOpp = false;
        }
    }
}
