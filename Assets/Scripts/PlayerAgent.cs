using System.Linq;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

[RequireComponent(typeof(PlayerMovement), typeof(Rigidbody))]
public class PlayerAgent : Agent
{
    [Header("참조")]
    public PlayerMovement movement;
    public EnemyShooting shooting;                 // 스윙 실행용
    public SwingZone overZone;                     // 오버 스윙 존 (Clear, Drop)
    public SwingZone underZone;                    // 언더 스윙 존 (Hairpin, Drive, Under)
    public Collider myCourtCollider;               // "MySide" 태그를 가진 콜라이더
    public ReinforcementLearningManager rl;        // courtHalfWidthX/Z 등 값 참조

    [Header("태그/이름")]
    public string shuttlecockTag = "Shuttlecock";  
    public string goalTag = "Goal";

    [Header("보상")]
    public float stepCenteringWeight = 0.3f;      // 낙하지점 근접 shaping 계수(프레임당)
    public float hitReward = 2.0f;                 // 셔틀 타격 성공 보상
    public float timePenalty = -0.0005f;           // 소량 지연 패널티
    public float wrongZoneSwingPenalty = -0.3f;    // 잘못된 존에서 스윙 선택 패널티
	public float outOfCourtPenalty = -0.005f;       // 내 코트 벗어남 패널티 (프레임당)
    public float centerProgressWeight = 0.1f;      // 낙하지점 근접 '진전' 보상 가중치
	public float swingAppropriateReward = 2.0f;    // 네트 거리 기준 적절한 스윙 보상

    [Header("에피소드 종료(랠리)")]
    public float landYThreshold = 0.7f;   // 셔틀 착지로 간주할 높이 
    public float postLandGrace = 0.0f;    // 착지 후 EndEpisode까지 추가 지연(0이면 즉시)
    GameObject _trackedShuttle;           // 이번 랠리 동안 추적할 셔틀
    bool _landedDetected;                 // 착지 감지 여부
    float _landedAt;                      // 착지 시각(Realtime)

	[Header("에피소드 로깅")]
	public bool logEpisodeRewards = true;             // 에피소드 리워드 요약 로그 출력 여부
	float _epTimePenaltySum = 0f;
	float _epCenteringSum = 0f;
	float _epOutOfCourtSum = 0f;
	float _epWrongZoneSum = 0f;
	float _epSwingAppropriateSum = 0f;
	float _epHitSum = 0f;
	int[] _epSwingExecCounts = new int[5];           // Clear, Drop, Hairpin, Drive, Under 실행 횟수
	int _epWrongZoneSelectCount = 0;
    float _prevCloseness = 0f;                       // 이전 프레임의 goal 근접도
	bool _overPrev = false, _underPrev = false;      // 이전 프레임 존 상태
	bool _overActiveLastFrame = false, _underActiveLastFrame = false; // 존 활성 상태(이전 프레임)
	bool _overSwungThisEntry = false, _underSwungThisEntry = false;   // 존 '이번 진입' 동안 스윙 실행 여부
	bool _combinedActiveLastFrame = false;           // 오버/언더 통합 활성 상태(이전 프레임)
	bool _swungThisCombinedEntry = false;            // 통합 존에 대한 '이번 진입' 동안 스윙 실행 여부
	[Header("디버그")]
	public bool logSwing = false;                      // 스윙 로그 출력 토글

    Rigidbody _rb;
    bool _hitGivenThisStep = false;                // 중복 지급 방지
    bool _episodeClosing;

    bool _hitRange = false;   // 셔틀이 타격 콜라이더 안에 있는지
    bool isMyTurn = false;    // 현재 내 턴 여부
    
    int _lastSwingAction = -1;  // 이전 스윙 액션 (중복 실행 방지)
    int _lastLoggedSwing = -1;  // 마지막으로 로그를 출력한 스윙


    public bool mySideIsPositiveZ = false;

	
	[Header("쿨타임")]
	public float swingCooldown = 0.5f;   // 스윙 시도 후 대기 시간(초)
	float _nextSwingTime = 0f;

    public void SetTurn(bool value)
    {
        isMyTurn = value;
    }

    // === 공통 상수/유틸 ===
    static readonly string[] SwingNames = { "Clear", "Drop", "Hairpin", "Drive", "Under" };

    bool IsOutsideCourtXZ(Vector3 p)
    {
        if (myCourtCollider == null) return false;
        Bounds b = myCourtCollider.bounds;
        return (p.x < b.min.x || p.x > b.max.x || p.z < b.min.z || p.z > b.max.z);
    }

    void ApplyWrongZonePenalty()
    {
        AddReward(wrongZoneSwingPenalty);
        _epWrongZoneSum += wrongZoneSwingPenalty;
        _epWrongZoneSelectCount++;
    }

	// === 헬퍼 메서드(중복 제거) ===
	bool IsOverSwingIndex(int s) { return s == 0 || s == 1; }
	bool IsUnderSwingIndex(int s) { return s == 2 || s == 3 || s == 4; }
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
	void LogSwingIfChanged(int swing, SwingZone.ZoneType activeZoneType)
	{
		if (!logSwing) return;
		if (swing == _lastLoggedSwing) return;
		if (swing < 0 || swing >= SwingNames.Length) return;
		Debug.Log($"[PlayerAgent] 스윙: {SwingNames[swing]} (존: {activeZoneType})");
		_lastLoggedSwing = swing;
	}
	bool TryProcessSwingChange(int swing, bool overHitRange, bool underHitRange, out SwingZone.ZoneType activeZoneType)
	{
		activeZoneType = SwingZone.ZoneType.Over;

		// 쿨타임 중이면 시도하지 않음
		if (Time.time < _nextSwingTime)
			return false;

		// 스윙존에 머무르는 동안은 1회만 스윙
		if ((overHitRange || underHitRange) && _swungThisCombinedEntry)
			return false;

		// 같은 존에 머무르는 동안은 1회만 스윙
		if (overHitRange && _overSwungThisEntry) return false;
		if (underHitRange && _underSwungThisEntry) return false;

		// 오버 존에서의 처리
		if (overHitRange && overZone != null)
		{
			if (IsOverSwingIndex(swing))
			{
				// 올바른 스윙 → 실행
				if (HasTargetsNowForSwing(swing))
				{
					activeZoneType = SwingZone.ZoneType.Over;
					LogSwingIfChanged(swing, activeZoneType);
					ExecuteSwingByIndex(swing);
					_swungThisCombinedEntry = true;
					_overSwungThisEntry = true;
					if (swing >= 0 && swing < _epSwingExecCounts.Length) _epSwingExecCounts[swing]++;
					_nextSwingTime = Time.time + swingCooldown;
					return true;
				}
			}
			else if (IsUnderSwingIndex(swing))
			{
				// 잘못된 스윙 → 패널티
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
					LogSwingIfChanged(swing, activeZoneType);
					ExecuteSwingByIndex(swing);
					_swungThisCombinedEntry = true;
					_underSwungThisEntry = true;
					if (swing >= 0 && swing < _epSwingExecCounts.Length) _epSwingExecCounts[swing]++;
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
			if (swing == 0 || swing == 3)
			{
			AddReward(swingAppropriateReward);
			_epSwingAppropriateSum += swingAppropriateReward;
			}
		}
		else if (distanceRatio < 0.4f)
		{
			if (swing == 1 || swing == 2 || swing == 4)
			{
			AddReward(swingAppropriateReward);
			_epSwingAppropriateSum += swingAppropriateReward;
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
        
        // "MySide" 태그를 가진 콜라이더 자동 찾기
        if (!myCourtCollider)
        {
            GameObject mySideObj = GameObject.FindGameObjectWithTag("MySide");
            if (mySideObj != null)
                myCourtCollider = mySideObj.GetComponent<Collider>();
        }

        if (movement) movement.trainingMode = true;

		// Shooter가 존 참조를 갖도록 보정 (추론 씬에서 누락될 수 있음)
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

        // trainingMode 확인 (에피소드마다 확인)
        if (movement) movement.trainingMode = true;

		// 랠리 상태 초기화
        _trackedShuttle = null;
        _landedDetected = false;
        _landedAt = 0f;
        _lastLoggedSwing = -1;  // 스윙 로그 초기화

		// 에피소드 리워드/카운트 초기화
		_epTimePenaltySum = 0f;
		_epCenteringSum = 0f;
		_epOutOfCourtSum = 0f;
		_epWrongZoneSum = 0f;
		_epSwingAppropriateSum = 0f;
		_epHitSum = 0f;
		_epSwingExecCounts = new int[5];
		_epWrongZoneSelectCount = 0;
        _prevCloseness = 0f;
		_overPrev = _underPrev = false;
		_lastSwingAction = -1;
		_nextSwingTime = 0f;
		_overActiveLastFrame = _underActiveLastFrame = false;
		_overSwungThisEntry = _underSwungThisEntry = false;
		_combinedActiveLastFrame = false;
		_swungThisCombinedEntry = false;
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
        int swing = (da.Length >= 1) ? Mathf.Clamp(da[0], 0, 4) : 0;  // 0~4 (5가지 스윙: Clear, Drop, Drive, Smash, Hairpin)
        
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
		bool combinedHitRange = _hitRange;
		if (combinedHitRange && !_combinedActiveLastFrame) _swungThisCombinedEntry = false;

		// 스윙존 진입 감지 시 해당 존의 스윙-1회 플래그 리셋
		if (overHitRange && !_overActiveLastFrame) _overSwungThisEntry = false;
		if (underHitRange && !_underActiveLastFrame) _underSwungThisEntry = false;

		// 이전 프레임 상태 기록(참고용)
		_overPrev = overHitRange;
		_underPrev = underHitRange;

		// 스윙 트리거: 쿨타임 기반 매 프레임 시도(조건 불충족이면 내부에서 무시)
		bool executedThisStep = false;
		SwingZone.ZoneType activeZoneType = SwingZone.ZoneType.Over;
		executedThisStep = TryProcessSwingChange(swing, overHitRange, underHitRange, out activeZoneType);
		_lastSwingAction = swing;

		// 현재 프레임의 존 활성 상태를 '이전 프레임'으로 기록
		_overActiveLastFrame = overHitRange;
		_underActiveLastFrame = underHitRange;
		_combinedActiveLastFrame = combinedHitRange;

        // ----- 보상(Reward) -----
		AddReward(timePenalty); // 소량 시간 패널티
		_epTimePenaltySum += timePenalty;
        
		// 내 코트 벗어남 패널티 (XZ 평면 기준)
		if (IsOutsideCourtXZ(transform.position))
		{
			AddReward(outOfCourtPenalty);
			_epOutOfCourtSum += outOfCourtPenalty;
		}

		// 네트까지의 거리 계산 (거리에 따라 적절한 스윙 타입 선택)
        float cw = rl ? rl.courtHalfWidthX : 11f;
        float cz = rl ? rl.courtHalfLengthZ : 20f;
        var goal = GameObject.FindGameObjectWithTag(goalTag);
		float netDistanceToTarget = 0f;
        bool goalValid = false;
        
        if (goal != null)
        {
			Vector3 goalPos = goal.transform.position;
			// 네트(z=0)까지의 절대 거리 사용
			netDistanceToTarget = Mathf.Abs(goalPos.z);
			float maxDistance = cz; // 네트에서 베이스라인까지 반코트 길이
            goalValid = true;
            
			// 거리에 따른 적절한 스윙 타입 보상 (스윙이 실제 실행된 프레임에만 1회 지급)
			// 멀리 쳐야 할 때(네트 거리 > 0.6*max): Clear(0), Drive(3) 적합
			// 가까이 쳐야 할 때(네트 거리 < 0.4*max): Drop(1), Hairpin(2), Under(4) 적합
			float distanceRatio = netDistanceToTarget / maxDistance;

            MaybeRewardAppropriateSwing(swing, executedThisStep, distanceRatio);
        }

        if (goalValid && goal && rl)
        {
            float gX = Mathf.Clamp(goal.transform.position.x, -cw, cw) / cw; // [-1,1]
            float gZ = Mathf.Clamp(goal.transform.position.z, -cz, cz) / cz; // [-1,1]
            float gOut = (Mathf.Abs(goal.transform.position.x / cw) > 1f ||
                          Mathf.Abs(goal.transform.position.z / cz) > 1f) ? 1f : 0f;

            // 관측-보상 정합성: Goal이 내 코트(XZ 기준)에 있으면 내 턴으로 간주
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
                // 내 턴 & 인코트일 때만 낙하지점 근접 shaping 부여
                Vector2 pN = new Vector2(
                    Mathf.Clamp(transform.position.x, -cw, cw) / cw,
                    Mathf.Clamp(transform.position.z, -cz, cz) / cz
                );
                Vector2 gN = new Vector2(gX, gZ);

                float dist = Vector2.Distance(pN, gN);                  // 0..~2.8284 (√8)
                float closeness = Mathf.Clamp01(1f - (dist / 2.8284f)); // 0..1
                // 가까워진 양(양수)일 때만 보상
                float delta = Mathf.Max(0f, closeness - _prevCloseness);
                float progR = centerProgressWeight * delta;
                AddReward(progR);
                _epCenteringSum += progR;
                _prevCloseness = closeness;
            }
        }

    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var ca = actionsOut.ContinuousActions;
        var da = actionsOut.DiscreteActions;

        ca[0] = Input.GetAxis("Horizontal");
        ca[1] = Input.GetAxis("Vertical");

        // 테스트용: 키 입력에 따라 스윙 종류 0~4 (5가지)
        // 1: Clear, 2: Drop, 3: Hairpin, 4: Drive, 5: Under
        if (Input.GetKey(KeyCode.Alpha1)) da[0] = 0;      // Clear
        else if (Input.GetKey(KeyCode.Alpha2)) da[0] = 1;  // Drop
        else if (Input.GetKey(KeyCode.Alpha3)) da[0] = 2;  // Hairpin
        else if (Input.GetKey(KeyCode.Alpha4)) da[0] = 3;  // Drive
        else if (Input.GetKey(KeyCode.Alpha5)) da[0] = 4;  // Under
        else da[0] = 0;  // 기본값: Clear

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
			_epHitSum += hitReward;
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
				// 에피소드 리워드 요약 로그
				if (logEpisodeRewards) LogEpisodeRewards();

				EndEpisode();
				// 다음 에피소드를 위해 상태 초기화
				_trackedShuttle = null;
				_landedDetected = false;
				_landedAt = 0f;
			}
        }
    }

	void LogEpisodeRewards()
	{
		float total = GetCumulativeReward();
		string swingCounts = string.Format("{0}:{1}, {2}:{3}, {4}:{5}, {6}:{7}, {8}:{9}",
			SwingNames[0], _epSwingExecCounts.Length > 0 ? _epSwingExecCounts[0] : 0,
			SwingNames[1], _epSwingExecCounts.Length > 1 ? _epSwingExecCounts[1] : 0,
			SwingNames[2], _epSwingExecCounts.Length > 2 ? _epSwingExecCounts[2] : 0,
			SwingNames[3], _epSwingExecCounts.Length > 3 ? _epSwingExecCounts[3] : 0,
			SwingNames[4], _epSwingExecCounts.Length > 4 ? _epSwingExecCounts[4] : 0
		);

		Debug.Log(
			$"[EpLog] total={total:F3} | time={_epTimePenaltySum:F3}, center={_epCenteringSum:F3}, outCourt={_epOutOfCourtSum:F3}, wrongZone={_epWrongZoneSum:F3} (cnt={_epWrongZoneSelectCount}), swingFit={_epSwingAppropriateSum:F3}, hit={_epHitSum:F3} | swings={{ {swingCounts} }}"
		);
	}
}
