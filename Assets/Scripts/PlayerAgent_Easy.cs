using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

[RequireComponent(typeof(PlayerMovement), typeof(Rigidbody))]
public class PlayerAgent_Easy : Agent
{
    [Header("참조")]
    public PlayerMovement movement;
    public EnemyShooting shooting;                 // 스윙 실행용
    public SwingZone overZone;                     // 오버 스윙 존 (Clear, Drop)
    public ReinforcementLearningManager rl;        // courtHalfWidthX/Z 등 값 참조
    public Collider myCourtCollider;               // "MySide" 태그를 가진 콜라이더

    [Header("태그/이름")]
    public string shuttlecockTag = "Shuttlecock";
    public string goalTag = "Goal";

    [Header("보상")]
    public float timePenalty = -0.0005f;           // 소량 지연 패널티
    public float swingReward = 1.0f;               // 스윙 성공 보상
    public float centerProgressWeight = 0.1f;      // 골 지점 추종 보상 가중치

    [Header("에피소드 종료(랠리)")]
    public float landYThreshold = 0.7f;            // 셔틀 착지로 간주할 높이
    public float postLandGrace = 0.0f;             // 착지 후 EndEpisode까지 추가 지연(0이면 즉시)
    GameObject _trackedShuttle;                    // 이번 랠리 동안 추적할 셔틀
    bool _landedDetected;                          // 착지 감지 여부
    float _landedAt;                               // 착지 시각(Realtime)

    Rigidbody _rb;
    bool _hitRange;                                // 셔틀이 타격 콜라이더 안에 있는지

    [Header("쿨타임")]
    public float swingCooldown = 0.5f;
    float _nextSwingTime;
    bool isMyTurn;
    float _prevCloseness;

    bool HasTargetsIn(SwingZone zone) => zone != null && zone.GetShuttlecocks().Count > 0;

    void ExecuteSwingByIndex(int swing)
    {
        switch (swing)
        {
            case 0:
                shooting.Clear();
                break;
            case 1:
                shooting.Drop();
                break;
        }
    }

    public override void Initialize()
    {
        if (!movement) movement = GetComponent<PlayerMovement>();
        if (!_rb) _rb = GetComponent<Rigidbody>();
        if (!rl) rl = FindObjectOfType<ReinforcementLearningManager>();

        if (!overZone)
        {
            SwingZone[] zones = GetComponentsInChildren<SwingZone>();
            foreach (var zone in zones)
            {
                if (zone.zoneType == SwingZone.ZoneType.Over)
                {
                    overZone = zone;
                    break;
                }
            }
        }

        if (movement) movement.trainingMode = true;

        if (shooting != null && shooting.overZone == null)
        {
            shooting.overZone = overZone;
        }
    }

    public override void OnEpisodeBegin()
    {
        if (_rb) _rb.velocity = Vector3.zero;
        if (movement) movement.SetMoveInput(Vector2.zero);

        _trackedShuttle = null;
        _landedDetected = false;
        _landedAt = 0f;
        _nextSwingTime = 0f;
        isMyTurn = false;
        _prevCloseness = 0f;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        float cw = rl ? rl.courtHalfWidthX : 11f;
        float cz = rl ? rl.courtHalfLengthZ : 20f;

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

        var sc = FindNearestShuttlecock();
        Vector3 rel = Vector3.zero;
        if (sc)
            rel = sc.transform.position - transform.position;

        sensor.AddObservation(Mathf.Clamp(rel.x / cw, -1f, 1f));
        sensor.AddObservation(Mathf.Clamp(rel.y / 10f, -1f, 1f));
        sensor.AddObservation(Mathf.Clamp(rel.z / cz, -1f, 1f));

        sensor.AddObservation(_hitRange ? 1f : 0f);
        sensor.AddObservation(Mathf.Clamp(transform.position.x, -cw, cw) / cw);
        sensor.AddObservation(Mathf.Clamp(transform.position.z, -cz, cz) / cz);
        sensor.AddObservation(isMyTurn ? 1f : 0f);
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        var ca = actions.ContinuousActions;
        var da = actions.DiscreteActions;

        Vector2 move = new Vector2(Mathf.Clamp(ca[0], -1f, 1f), Mathf.Clamp(ca[1], -1f, 1f));
        int swing = (da.Length >= 1) ? Mathf.Clamp(da[0], 0, 1) : 0;  // 0=Clear, 1=Drop

        if (movement == null)
        {
            movement = GetComponent<PlayerMovement>();
            if (movement == null)
            {
                Debug.LogError("[PlayerAgent_Easy] PlayerMovement 컴포넌트를 찾을 수 없습니다!");
                return;
            }
        }

        movement.SetMoveInput(move);

        bool overHitRange = HasTargetsIn(overZone);
        _hitRange = overHitRange;

        bool executedThisStep = false;
        if (overHitRange && shooting != null && Time.time >= _nextSwingTime)
        {
            ExecuteSwingByIndex(swing);
            _nextSwingTime = Time.time + swingCooldown;
            executedThisStep = true;
        }

        AddReward(timePenalty);

        if (executedThisStep)
        {
            AddReward(swingReward);
        }

        float cw = rl ? rl.courtHalfWidthX : 11f;
        float cz = rl ? rl.courtHalfLengthZ : 20f;
        var goal = GameObject.FindGameObjectWithTag(goalTag);
        bool goalValid = false;

        if (goal)
        {
            goalValid = true;

            if (myCourtCollider != null)
            {
                Bounds b = myCourtCollider.bounds;
                Vector3 gp = goal.transform.position;
                isMyTurn = (gp.x >= b.min.x && gp.x <= b.max.x && gp.z >= b.min.z && gp.z <= b.max.z);
            }
            else
            {
                // 기본값: goal이 내 z 구간에 있으면 내 턴으로 가정
                float gz = goal.transform.position.z;
                isMyTurn = rl ? (rl.courtHalfLengthZ >= 0f && gz <= 0f) : (gz <= 0f);
            }
        }

        if (goalValid && rl)
        {
            float gX = Mathf.Clamp(goal.transform.position.x, -cw, cw) / cw;
            float gZ = Mathf.Clamp(goal.transform.position.z, -cz, cz) / cz;
            float gOut = (Mathf.Abs(goal.transform.position.x / cw) > 1f ||
                          Mathf.Abs(goal.transform.position.z / cz) > 1f) ? 1f : 0f;

            if (gOut < 0.5f && isMyTurn)
            {
                Vector2 pN = new Vector2(
                    Mathf.Clamp(transform.position.x, -cw, cw) / cw,
                    Mathf.Clamp(transform.position.z, -cz, cz) / cz
                );
                Vector2 gN = new Vector2(gX, gZ);

                float dist = Vector2.Distance(pN, gN);
                float closeness = Mathf.Clamp01(1f - (dist / 2.8284f));
                float delta = Mathf.Max(0f, closeness - _prevCloseness);
                float progR = centerProgressWeight * delta;
                AddReward(progR);
                _prevCloseness = closeness;
            }
            else
            {
                Vector2 pN = new Vector2(
                    Mathf.Clamp(transform.position.x, -cw, cw) / cw,
                    Mathf.Clamp(transform.position.z, -cz, cz) / cz
                );
                Vector2 gN = new Vector2(gX, gZ);
                float dist = Vector2.Distance(pN, gN);
                _prevCloseness = Mathf.Clamp01(1f - (dist / 2.8284f));
            }
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
        if (_trackedShuttle == null)
        {
            _trackedShuttle = FindNearestShuttlecock();
        }

        if (!_landedDetected && _trackedShuttle != null)
        {
            if (_trackedShuttle.transform.position.y <= landYThreshold)
            {
                _landedDetected = true;
                _landedAt = Time.realtimeSinceStartup;
            }
        }

        if (_landedDetected)
        {
            if (postLandGrace > 0f && Time.realtimeSinceStartup - _landedAt < postLandGrace)
                return;

            bool trackedGone = _trackedShuttle == null || _trackedShuttle.Equals(null);
            if (!trackedGone)
            {
                trackedGone = (_trackedShuttle == null);
            }

            if (trackedGone)
            {
                EndEpisode();
                _trackedShuttle = null;
                _landedDetected = false;
                _landedAt = 0f;
            }
        }
    }
}
