using System.Linq;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

[RequireComponent(typeof(EnemyMovement), typeof(Rigidbody))]
public class PlayerAgent : Agent
{
    [Header("Refs")]
    public EnemyMovement movement;
    public EnemyShooting shooting;                // (선택) 스윙 실행용
    public ReinforcementLearningManager rl;        // courtHalfWidthX/Z 등 값 참조

    [Header("Tags/Names")]
    public string shuttlecockTag = "Shuttlecock";  // RLManager와 동일하게 유지
    public string goalTag = "Goal";

    [Header("Rewards")]
    public float stepCenteringWeight = 0.01f;      // 낙하지점 근접 shaping 계수(프레임당)
    public float hitReward = 2.0f;                 // 셔틀 타격 성공 보상
    public float timePenalty = -0.0005f;           // 소량 지연 패널티

    [Header("Episode End (Rally)")]
    public float landYThreshold = 0.7f;   // 셔틀 '착지'로 간주할 높이
    public float postLandGrace = 0.0f;    // 착지 후 EndEpisode까지 추가 지연(0이면 즉시)
    GameObject _trackedShuttle;           // 이번 랠리 동안 추적할 셔틀
    bool _landedDetected;                 // 착지 감지 여부
    float _landedAt;                      // 착지 시각(Realtime)

    [Header("Court Area")]
    public Collider myCourtArea; // 내 코트 

    Rigidbody _rb;
    bool _hitGivenThisStep = false;                // 중복 지급 방지
    bool _episodeClosing;



    public override void Initialize()
    {
        if (!movement) movement = GetComponent<EnemyMovement>();
        if (!_rb) _rb = GetComponent<Rigidbody>();
        if (!rl) rl = FindObjectOfType<ReinforcementLearningManager>();

        if (movement) movement.trainingMode = true;
    }

    bool IsInsideCollider(Collider col, Vector3 point)
    {
        if (!col || col.Equals(null)) return false;
        // 콜라이더 내부면 ClosestPoint가 point 자체(또는 매우 근접)를 반환
        Vector3 c = col.ClosestPoint(point);
        return (c - point).sqrMagnitude < 1e-6f;
    }

    public override void OnEpisodeBegin()
    {
        // 간단 초기화 (필요시 위치/회전 리셋은 프로젝트 규칙에 맞춰 추가)
        _rb.velocity = Vector3.zero;
        movement.SetMoveInput(Vector2.zero);

        // ★ 랠리 상태 초기화
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

        // 1) Player 정규화 위치
        Vector3 p = transform.position;
        float pX = Mathf.Clamp(p.x, -cw, cw) / cw;   // [-1,1]
        float pZ = Mathf.Clamp(p.z, -cz, cz) / cz;   // [-1,1]
        sensor.AddObservation(pX);
        sensor.AddObservation(pZ);

        // 2) Goal (예상 낙하지점)
        var goal = GameObject.FindGameObjectWithTag(goalTag);
        float gXDisp = 0f, gZDisp = 0f, gOut = 0f;
        if (goal)
        {
            Vector3 g = goal.transform.position;

            // 표시용 정규화 규칙: z는 코트 방향에 따라 0..1 / -1..0로 분기
            gXDisp = Mathf.Clamp(g.x, -cw, cw) / cw;
            if (g.z >= 0f) gZDisp = Mathf.Clamp(g.z, 0f, cz) / cz;
            else gZDisp = Mathf.Clamp(g.z, -cz, 0f) / cz;

            // 코트 밖 판정은 클램프 없이 계산
            float gXRaw = g.x / cw;
            float gZRaw = g.z / cz;
            gOut = (Mathf.Abs(gXRaw) > 1f || Mathf.Abs(gZRaw) > 1f) ? 1f : 0f;
        }
        sensor.AddObservation(gXDisp);
        sensor.AddObservation(gZDisp);
        sensor.AddObservation(gOut);

        // 3) 셔틀콕 탐색: 가장 가까운 1개
        var sc = FindNearestShuttlecock();
        Vector3 rel = Vector3.zero;
        float scxNorm = 0f, sczNorm = 0f, speed = 0f;

        if (sc)
        {
            rel = sc.transform.position - transform.position;

            Vector3 scp = sc.transform.position;
            scxNorm = Mathf.Clamp(scp.x, -cw, cw) / cw;
            if (scp.z >= 0f) sczNorm = Mathf.Clamp(scp.z, 0f, cz) / cz;
            else sczNorm = Mathf.Clamp(scp.z, -cz, 0f) / cz;

            var scRb = sc.GetComponent<Rigidbody>();
            speed = scRb ? scRb.velocity.magnitude : 0f;
        }

        // 상대 위치는 크기 스케일링(대략 코트 하프 크기/높이로 정규화)
        sensor.AddObservation(Mathf.Clamp(rel.x / cw, -1f, 1f));
        sensor.AddObservation(Mathf.Clamp(rel.y / 10f, -1f, 1f));
        sensor.AddObservation(Mathf.Clamp(rel.z / cz, -1f, 1f));

        sensor.AddObservation(scxNorm);
        sensor.AddObservation(sczNorm);
        sensor.AddObservation(Mathf.Clamp(speed / 50f, 0f, 1f)); // 속도 상한 가정
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

        //if (da[0] == 1) movement.Jump();
        if (da[0] == 1 && shooting) shooting.OverStrong();
        if (da[0] == 2 && shooting) shooting.OverWeak();
        if (da[0] == 3 && shooting) shooting.UnderStrong();
        if (da[0] == 4 && shooting) shooting.UnderWeak();

        // ----- 보상(Reward) -----
        AddReward(timePenalty); // 소량 시간 패널티

        // (1) 낙하지점 근접 보상(유동형 shaping)
        var goal = GameObject.FindGameObjectWithTag(goalTag);
        if (goal && rl)
        {
            // 내 코트 콜라이더가 설정되어 있고, Goal이 그 안에 있을 때만 보상
            bool goalInMyCourt = myCourtArea ? IsInsideCollider(myCourtArea, goal.transform.position) : true;
            if (goalInMyCourt)
            {
                // 플레이어/Goal을 동일 축으로 정규화한 뒤 2D 거리
                float cw = rl.courtHalfWidthX;
                float cz = rl.courtHalfLengthZ;

                Vector2 pN = new Vector2(
                    Mathf.Clamp(transform.position.x, -cw, cw) / cw,
                    Mathf.Clamp(transform.position.z, -cz, cz) / cz
                );

                Vector2 gN = new Vector2(
                    Mathf.Clamp(goal.transform.position.x, -cw, cw) / cw,
                    Mathf.Clamp(goal.transform.position.z, -cz, cz) / cz
                );

                float dist = Vector2.Distance(pN, gN);                 // 0..~√2
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
        da[0] = Input.GetKey(KeyCode.Space) ? 1 : 0;
        da[1] = Input.GetKey(KeyCode.Q) ? 1 : 0;
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
