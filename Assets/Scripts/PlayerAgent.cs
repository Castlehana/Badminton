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
    public EnemyShooting shooting;                 // 스윙 실행용
    public ReinforcementLearningManager rl;        // courtHalfWidthX/Z 등 값 참조

    [Header("Tags/Names")]
    public string shuttlecockTag = "Shuttlecock";  
    public string goalTag = "Goal";

    [Header("Rewards")]
    public float stepCenteringWeight = 0.01f;      // 낙하지점 근접 shaping 계수(프레임당)
    public float hitReward = 2.0f;                 // 셔틀 타격 성공 보상
    public float timePenalty = -0.0005f;           // 소량 지연 패널티

    [Header("Episode End (Rally)")]
    public float landYThreshold = 0.7f;   // 셔틀 착지로 간주할 높이 
    public float postLandGrace = 0.0f;    // 착지 후 EndEpisode까지 추가 지연(0이면 즉시)
    GameObject _trackedShuttle;           // 이번 랠리 동안 추적할 셔틀
    bool _landedDetected;                 // 착지 감지 여부
    float _landedAt;                      // 착지 시각(Realtime)

    Rigidbody _rb;
    bool _hitGivenThisStep = false;                // 중복 지급 방지
    bool _episodeClosing;

    bool _hitRange = false;   // 셔틀이 타격 콜라이더 안에 있는지
    bool isMyTurn = false;    // 현재 내 턴 여부


    public bool mySideIsPositiveZ = false;

    public void SetHitRange(bool value)
    {
        _hitRange = value;
    }

    public void SetTurn(bool value)
    {
        isMyTurn = value;
    }


    public override void Initialize()
    {
        if (!movement) movement = GetComponent<PlayerMovement>();
        if (!_rb) _rb = GetComponent<Rigidbody>();
        if (!rl) rl = FindObjectOfType<ReinforcementLearningManager>();

        if (movement) movement.trainingMode = true;
    }

    /*
    bool IsInsideCollider(Collider col, Vector3 point)
    {
        if (!col || col.Equals(null)) return false;
        // 콜라이더 내부면 ClosestPoint가 point 자체(또는 매우 근접)를 반환
        Vector3 c = col.ClosestPoint(point);
        return (c - point).sqrMagnitude < 1e-6f;
    }*/

    public override void OnEpisodeBegin()
    {
        // 간단 초기화 (필요시 위치/회전 리셋은 프로젝트 규칙에 맞춰 추가)
        _rb.velocity = Vector3.zero;
        movement.SetMoveInput(Vector2.zero);

        // 랠리 상태 초기화
        _trackedShuttle = null;
        _landedDetected = false;
        _landedAt = 0f;
    }

    // === 관측(OBS) ===
    // ReinforcementLearningManager가 다루는 값들 기반:
    // - Player 정규화 위치 x,z (courtHalfWidthX/Z 기준)
    // - Goal 정규화 x,z (표시용 규칙과 동일)
    // - Goal 코트 밖 여부 isOutOfCourt (float)
    // - "플레이어 기준" 셔틀콕 상대 위치 x,y,z
    // - 셔틀콕의 네트 기준 정규화 x,z
    // - 셔틀콕 속력(speed)
    //
    // 총 11 floats
    public override void CollectObservations(VectorSensor sensor)
    {
        float cw = rl ? rl.courtHalfWidthX : 11f;
        float cz = rl ? rl.courtHalfLengthZ : 20f;

        // ① Goal 낙하 지점 예측 (x,z 정규화 + 코트밖 여부)
        var goal = GameObject.FindGameObjectWithTag(goalTag);
        float gXDisp = 0f, gZDisp = 0f, gOut = 0f;
        if (goal)
        {
            Vector3 g = goal.transform.position;
            gXDisp = Mathf.Clamp(g.x, -cw, cw) / cw;
            gZDisp = Mathf.Clamp(g.z, -cz, cz) / cz;

            float gXRaw = g.x / cw;
            float gZRaw = g.z / cz;
            gOut = (Mathf.Abs(gXRaw) > 1f || Mathf.Abs(gZRaw) > 1f) ? 1f : 0f;
        }
        sensor.AddObservation(gXDisp);
        sensor.AddObservation(gZDisp);
        sensor.AddObservation(gOut);

        // ② 셔틀콕 상대 위치 (x,y,z)
        var sc = FindNearestShuttlecock();
        Vector3 rel = Vector3.zero;
        if (sc)
            rel = sc.transform.position - transform.position;

        sensor.AddObservation(Mathf.Clamp(rel.x / cw, -1f, 1f));
        sensor.AddObservation(Mathf.Clamp(rel.y / 10f, -1f, 1f));
        sensor.AddObservation(Mathf.Clamp(rel.z / cz, -1f, 1f));

        // ③ 셔틀콕이 타격 범위 안에 있는지 (0 or 1)
        float hitPossible = _hitRange ? 1f : 0f;
        sensor.AddObservation(hitPossible);

        // ④ 네트 중심(z=0)으로부터의 거리 (0~1 정규화)
        float netDistance = Mathf.Abs(transform.position.z) / cz;
        sensor.AddObservation(netDistance);

        // ⑤ 코트 중심(x,z) 상대 거리 (정규화)
        float centerX = Mathf.Clamp(transform.position.x, -cw, cw) / cw;
        float centerZ = Mathf.Clamp(transform.position.z, -cz, cz) / cz;
        sensor.AddObservation(centerX);
        sensor.AddObservation(centerZ);

        // ⑥ 턴 정보 (내 턴 [1,0], 상대 턴 [0,1])
        float myTurn = isMyTurn ? 1f : 0f;
        float oppTurn = isMyTurn ? 0f : 1f;
        sensor.AddObservation(myTurn);
        sensor.AddObservation(oppTurn);
    }

    // === 행동 적용 ===
    // 연속 2 (move XZ) + 이산 4 (오버2, 언더2)
    public override void OnActionReceived(ActionBuffers actions)
    {
        _hitGivenThisStep = false;

        var ca = actions.ContinuousActions;
        var da = actions.DiscreteActions;

        Vector2 move = new Vector2(Mathf.Clamp(ca[0], -1f, 1f), Mathf.Clamp(ca[1], -1f, 1f));
        movement.SetMoveInput(move);

        int swing = (da.Length >= 1) ? da[0] + 1 : 1;  // 1~4로 변환


        // 스윙 실행
        switch (swing)
        {
            case 1: if (shooting) shooting.OverStrong(); break;
            case 2: if (shooting) shooting.OverWeak(); break;
            case 3: if (shooting) shooting.UnderStrong(); break;
            case 4: if (shooting) shooting.UnderWeak(); break;
        }

        // ----- 보상(Reward) -----
        AddReward(timePenalty); // 소량 시간 패널티

        // 네트 기준 거리 (0~1)
        float cw = rl ? rl.courtHalfWidthX : 11f;
        float cz = rl ? rl.courtHalfLengthZ : 20f;
        float netDist = Mathf.Abs(transform.position.z) / cz; // 0~1 정규화

        // 🔹 스윙 보상
        // 네트에 가까울수록 strong, 멀수록 weak이 적절함.
        if (netDist <= 0.4f)
        {
            if (swing == 1 || swing == 3)
            {
                AddReward(0.5f);
                Debug.Log($"[Reward] Strong swing success near net (dist={netDist:F2})");
            }
        }
        else
        {
            // 원거리 weak 스윙이면 큰 보상
            if (swing == 2 || swing == 4)
                AddReward(0.5f);
        }

        var goal = GameObject.FindGameObjectWithTag(goalTag);
        if (goal && rl)
        {
            float gX = Mathf.Clamp(goal.transform.position.x, -cw, cw) / cw; // [-1,1]
            float gZ = Mathf.Clamp(goal.transform.position.z, -cz, cz) / cz; // [-1,1]
            float gOut = (Mathf.Abs(goal.transform.position.x / cw) > 1f ||
                          Mathf.Abs(goal.transform.position.z / cz) > 1f) ? 1f : 0f;

            // 내 코트 판정 - Z<0이 내 코트
            const float Z_MARGIN = 0.05f; 
            bool goalOnMySide = mySideIsPositiveZ ? (gZ >= Z_MARGIN) : (gZ <= -Z_MARGIN);

            if (gOut < 0.5f && goalOnMySide)
            {
                // 내 코트 & 인코트일 때만 낙하지점 근접 shaping 부여
                Vector2 pN = new Vector2(
                    Mathf.Clamp(transform.position.x, -cw, cw) / cw,
                    Mathf.Clamp(transform.position.z, -cz, cz) / cz
                );
                Vector2 gN = new Vector2(gX, gZ);

                float dist = Vector2.Distance(pN, gN);                  // 0..~√2
                float closeness = Mathf.Clamp01(1f - (dist / 1.4142f)); // 0..1
                AddReward(stepCenteringWeight * closeness);
            }
        }

    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var ca = actionsOut.ContinuousActions;
        var da = actionsOut.DiscreteActions;

        ca[0] = Input.GetAxis("Horizontal");
        ca[1] = Input.GetAxis("Vertical");

        // 테스트용: 키 입력에 따라 스윙 종류 1~4
        if (Input.GetKey(KeyCode.Alpha1)) da[0] = 0;
        else if (Input.GetKey(KeyCode.Alpha2)) da[0] = 1;
        else if (Input.GetKey(KeyCode.Alpha3)) da[0] = 2;
        else if (Input.GetKey(KeyCode.Alpha4)) da[0] = 3;
        else da[0] = 0;

    }


    // 셔틀콕 충돌 감지 → 보상
    void OnCollisionEnter(Collision collision)
    {
        if (_hitGivenThisStep) return;
        if (collision.gameObject.CompareTag(shuttlecockTag))
        {
            // Shuttlecock 쪽에서 Player의 SphereCollider와만 충돌 허용되므로
            // 이 이벤트가 오면 "타격"으로 간주 가능
            AddReward(hitReward);
            _hitGivenThisStep = true;
        }
    }

    GameObject FindNearestShuttlecock()
    {
        var all = GameObject.FindGameObjectsWithTag(shuttlecockTag);
        if (all == null || all.Length == 0) return null;
        var me = transform.position;
        GameObject best = null;
        float bestD = float.MaxValue;
        foreach (var go in all)
        {
            float d = (go.transform.position - me).sqrMagnitude;
            if (d < bestD) { bestD = d; best = go; }
        }
        return best;
    }
    void LateUpdate()
    {
        // 1) 이번 랠리의 셔틀 지정 (없으면 가장 가까운 셔틀을 하나 잡아둠)
        if (_trackedShuttle == null)
        {
            _trackedShuttle = FindNearestShuttlecock();
        }

        // 2) 착지 감지: 추적 중인 셔틀의 y가 landYThreshold 이하로 내려오면 '착지'로 간주
        if (!_landedDetected && _trackedShuttle != null)
        {
            if (_trackedShuttle.transform.position.y <= landYThreshold)
            {
                _landedDetected = true;
                _landedAt = Time.realtimeSinceStartup;
            }
        }

        // 3) 에피소드 종료: 착지를 감지한 이후, 셔틀이 Destroy되어 "씬에 사라진" 순간 EndEpisode()
        //    (Shooter가 1초 뒤 새 셔틀을 스폰하기 전에 종료됨)
        if (_landedDetected)
        {
            // postLandGrace(지연)가 있으면 그만큼 기다렸다가 판단
            if (postLandGrace > 0f && Time.realtimeSinceStartup - _landedAt < postLandGrace)
                return;

            // 현재 씬에 셔틀이 더 이상 없으면 에피소드 종료
            // (혹시 다른 셔틀이 떠다니는 씬이라면, _trackedShuttle만 사라졌는지로 판단)
            bool trackedGone = _trackedShuttle == null || _trackedShuttle.Equals(null);
            if (!trackedGone)
            {
                // Destroy 직전 한 프레임에서 FindNearest가 null을 줄 수 있으니,
                // 안전하게 실제 존재 여부를 다시 검사
                trackedGone = (_trackedShuttle == null);
            }

            if (trackedGone)
            {
                EndEpisode();
                // 다음 에피소드를 위해 상태 초기화
                _trackedShuttle = null;
                _landedDetected = false;
                _landedAt = 0f;
            }
        }
    }
}
