using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;


[RequireComponent(typeof(PlayerMovement), typeof(Rigidbody))]
public class PlayerAgent_middle : Agent
{
    [Header("참조")]
    public PlayerMovement movement;
    public EnemyShooting shooting;                 // 스윙 실행용
    public SwingZone overZone;                     // 오버 스윙 존 (Clear, Drop, Drive)
    public SwingZone underZone;                    // 언더 스윙 존 (Hairpin, Under)
    public Collider myCourtCollider;               // "MySide" 태그를 가진 콜라이더
    public ReinforcementLearningManager rl;        // courtHalfWidthX/Z 등 값 참조

    [Header("태그/이름")]
    public string shuttlecockTag = "Shuttlecock";  
    public string goalTag = "Goal";

    [Header("보상")]
    public float timePenalty = -0.0005f;           // 소량 지연 패널티
    public float wrongZoneSwingPenalty = -0.5f;    // 잘못된 존에서 스윙 선택 패널티
    public float centerProgressWeight = 0.1f;      // 낙하지점 근접 '진전' 보상 가중치
    public float swingAppropriateReward = 1.0f;    // 네트 거리 기준 적절한 스윙 보상

    [Header("에피소드 종료(랠리)")]
    public float landYThreshold = 0.7f;   // 셔틀 착지로 간주할 높이 
    public float postLandGrace = 0.0f;    // 착지 후 EndEpisode까지 추가 지연(0이면 즉시)
    GameObject _trackedShuttle;           // 이번 랠리 동안 추적할 셔틀
    bool _landedDetected;                 // 착지 감지 여부
    float _landedAt;                      // 착지 시각(Realtime)

    Rigidbody _rb;

    bool _hitRange = false;   // 셔틀이 타격 콜라이더 안에 있는지
    bool isMyTurn = false;    // 현재 내 턴 여부

    public bool mySideIsPositiveZ = false;
	
	[Header("쿨타임")]
	public float swingCooldown = 0.5f;
	float _nextSwingTime = 0f;

    public void SetTurn(bool value)
    {
        isMyTurn = value;
    }

    float _prevCloseness = 0f;

    void ApplyWrongZonePenalty()
    {
        AddReward(wrongZoneSwingPenalty);
    }

	// === 헬퍼 메서드===
    bool IsOverSwingIndex(int s) { return s == 0 || s == 1 || s == 3; }
    bool IsUnderSwingIndex(int s) { return s == 2 || s == 4; }
	bool HasTargetsIn(SwingZone zone) { return zone != null && zone.GetShuttlecocks().Count > 0; }
	bool HasTargetsNowForSwing(int swing)
	{
		if (IsOverSwingIndex(swing)) return HasTargetsIn(overZone);
		if (IsUnderSwingIndex(swing)) return HasTargetsIn(underZone);
		return false;
	}
	void ExecuteSwingByIndex(int swing)
	{
		switch (swing)
		{
			case 0: shooting.Clear(); break;
			case 1: shooting.Drop(); break;
			case 2: shooting.Hairpin(); break;
			case 3: shooting.Drive(); break;
			case 4: shooting.Under(); break;
			default: break;
		}
	}

	bool TryProcessSwingChange(int swing, bool overHitRange, bool underHitRange, out SwingZone.ZoneType activeZoneType)
	{
		activeZoneType = SwingZone.ZoneType.Over;

		// 쿨타임 중이면 시도하지 않음
		if (Time.time < _nextSwingTime)
			return false;

		// 오버 존에서의 처리
		if (overHitRange && overZone != null)
		{
			if (IsOverSwingIndex(swing))
			{
				// 올바른 스윙은 실행
				if (HasTargetsNowForSwing(swing))
				{
					activeZoneType = SwingZone.ZoneType.Over;
					ExecuteSwingByIndex(swing);
					_nextSwingTime = Time.time + swingCooldown;
					return true;
				}
			}
			else if (IsUnderSwingIndex(swing))
			{
				// 잘못된 스윙은 패널티
				ApplyWrongZonePenalty();
				return false;
			}
		}

		// 언더 존에서의 처리
		if (underHitRange && underZone != null)
		{
			if (IsUnderSwingIndex(swing))
			{
				// 올바른 스윙 → 실행
				if (HasTargetsNowForSwing(swing))
				{
					activeZoneType = SwingZone.ZoneType.Under;
					ExecuteSwingByIndex(swing);
					_nextSwingTime = Time.time + swingCooldown;
					return true;
				}
			}
			else if (IsOverSwingIndex(swing))
			{
				// 잘못된 스윙 → 패널티
				ApplyWrongZonePenalty();
				return false;
			}
		}

		return false;
	}
	void MaybeRewardAppropriateSwing(int swing, bool executedThisStep, float distanceRatio)
	{
		if (!executedThisStep) return;
		if (distanceRatio > 0.6f)
		{
			if (swing == 0 || swing == 3 || swing == 4)
			{
				AddReward(swingAppropriateReward);
			}
		}
		else if (distanceRatio < 0.4f)
		{
			if (swing == 1 || swing == 2)
			{
				AddReward(swingAppropriateReward);
			}
		}
	}


    public override void Initialize()
    {
        if (!movement) movement = GetComponent<PlayerMovement>();
        if (!_rb) _rb = GetComponent<Rigidbody>();
        if (!rl) rl = FindObjectOfType<ReinforcementLearningManager>();
        
        // 오버/언더 존 자동 찾기
        if (!overZone || !underZone)
        {
            SwingZone[] zones = GetComponentsInChildren<SwingZone>();
            foreach (var zone in zones)
            {
                if (zone.zoneType == SwingZone.ZoneType.Over && !overZone)
                    overZone = zone;
                else if (zone.zoneType == SwingZone.ZoneType.Under && !underZone)
                    underZone = zone;
            }
        }

        if (movement) movement.trainingMode = true;

		// Shooter가 존 참조를 갖도록 보정
		if (shooting != null)
		{
			if (shooting.overZone == null) shooting.overZone = overZone;
			if (shooting.underZone == null) shooting.underZone = underZone;
		}
    }

    public override void OnEpisodeBegin()
    {
        // 간단 초기화
        _rb.velocity = Vector3.zero;
        movement.SetMoveInput(Vector2.zero);

		_prevCloseness = 0f;

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
    // 연속 2 (move XZ) + 이산 5 (오버3, 언더2)
    public override void OnActionReceived(ActionBuffers actions)
    {
        var ca = actions.ContinuousActions;
        var da = actions.DiscreteActions;

        Vector2 move = new Vector2(Mathf.Clamp(ca[0], -1f, 1f), Mathf.Clamp(ca[1], -1f, 1f));
        int swing = (da.Length >= 1) ? Mathf.Clamp(da[0], 0, 4) : 0;  // 0~4 (5가지 스윙)
        
        // movement 컴포넌트 null 체크 및 초기화
        if (movement == null)
        {
            movement = GetComponent<PlayerMovement>();
            if (movement == null)
            {
                Debug.LogError("[PlayerAgent] PlayerMovement 컴포넌트를 찾을 수 없습니다!");
                return;
            }
        }
        
        movement.SetMoveInput(move);

        // 오버/언더 존에서 스윙 가능 여부 확인
        bool overHitRange = overZone != null && overZone.GetShuttlecocks().Count > 0;
        bool underHitRange = underZone != null && underZone.GetShuttlecocks().Count > 0;
        _hitRange = overHitRange || underHitRange;

		// 스윙 트리거
		bool executedThisStep = false;
		SwingZone.ZoneType activeZoneType = SwingZone.ZoneType.Over;
		executedThisStep = TryProcessSwingChange(swing, overHitRange, underHitRange, out activeZoneType);

        // ----- 보상(Reward) -----
		AddReward(timePenalty);
        
		// 네트까지의 거리 계산 (거리에 따라 적절한 스윙 타입 선택)
        float cw = rl ? rl.courtHalfWidthX : 11f;
        float cz = rl ? rl.courtHalfLengthZ : 20f;
        var goal = GameObject.FindGameObjectWithTag(goalTag);
        bool goalValid = false;
        
        if (goal != null)
        {
			// 플레이어 기준 네트 거리 사용
			float maxDistance = cz; // 네트에서 베이스라인까지 반코트 길이
			float playerNetDistance = Mathf.Abs(transform.position.z);
			float distanceRatio = maxDistance > 0f ? Mathf.Clamp01(playerNetDistance / maxDistance) : 0f;
            goalValid = true;
            
			// 거리에 따른 적절한 스윙 타입 보상
			// 멀리 쳐야 할 때(네트 거리 > 0.6*max): Clear(0), Drive(3), Under(4) 적합
			// 가까이 쳐야 할 때(네트 거리 < 0.4*max): Drop(1), Hairpin(2) 적합

            MaybeRewardAppropriateSwing(swing, executedThisStep, distanceRatio);
        }

        if (goalValid && goal && rl)
        {
            float gX = Mathf.Clamp(goal.transform.position.x, -cw, cw) / cw; // [-1,1]
            float gZ = Mathf.Clamp(goal.transform.position.z, -cz, cz) / cz; // [-1,1]
            float gOut = (Mathf.Abs(goal.transform.position.x / cw) > 1f ||
                          Mathf.Abs(goal.transform.position.z / cz) > 1f) ? 1f : 0f;

            //  Goal이 내 코트(XZ 기준)에 있으면 내 턴으로 간주
            if (myCourtCollider != null)
            {
                Bounds b = myCourtCollider.bounds;
                Vector3 gp = goal.transform.position;
                bool goalOnMySideXZ = (gp.x >= b.min.x && gp.x <= b.max.x && gp.z >= b.min.z && gp.z <= b.max.z);
                isMyTurn = goalOnMySideXZ;
            }

            // 내 턴일 때 인코트 목표 지점을 따라가도록 shaping
            if (gOut < 0.5f && isMyTurn)
            {
                Vector2 pN = new Vector2(
                    Mathf.Clamp(transform.position.x, -cw, cw) / cw,
                    Mathf.Clamp(transform.position.z, -cz, cz) / cz
                );
                Vector2 gN = new Vector2(gX, gZ);

                float dist = Vector2.Distance(pN, gN);                  // 0..~2.8284 (√8)
                float closeness = Mathf.Clamp01(1f - (dist / 2.8284f)); // 0..1
                float delta = Mathf.Max(0f, closeness - _prevCloseness);
                float progR = centerProgressWeight * delta;
                AddReward(progR);
                _prevCloseness = closeness;
            }
            else
            {
                // 조건이 아니면 현재 근접도를 기록만 함
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
