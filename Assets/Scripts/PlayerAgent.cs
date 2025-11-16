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
    public SwingZone SmashDetect;          // 스매시
    public Collider myCourtCollider;               // "MySide" 태그를 가진 콜라이더
    public ReinforcementLearningManager rl;        // courtHalfWidthX/Z 등 값 참조


    [Header("태그/이름")]
    public string shuttlecockTag = "Shuttlecock";  
    public string goalTag = "Goal";


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


    bool smashMode = false;   // 점프 중 스매시 활성화 모드

    public bool forceSmashTest = true;

    float prevDropDist = 5f;

    // ===== 평가 지표 =====
    float prevGoalDist = 999f;   // 이전 스텝의 goal 거리
    float goalDistSum = 0f;      // 에피소드 동안의 거리 변화 누적
    int goalDistSteps = 0;       // 거리 측정한 총 스텝 수

    // 스윙 기록
    int[] swingHistory = new int[6]; // Clear, Drop, Hairpin, Drive, Under, Smash 카운트

    public bool mySideIsPositiveZ = false;

	
	[Header("쿨타임")]
	public float swingCooldown = 0.5f;   // 스윙 시도 후 대기 시간(초)
	float _nextSwingTime = 0f;

    public void SetTurn(bool value)
    {
        isMyTurn = value;
    }

    // === 공통 상수/유틸 ===
    static readonly string[] SwingNames = { "Clear", "Drop", "Hairpin", "Drive", "Under", "Smash"};

    bool IsOutsideCourtXZ(Vector3 p)
    {
        if (myCourtCollider == null) return false;
        Bounds b = myCourtCollider.bounds;
        return (p.x < b.min.x || p.x > b.max.x || p.z < b.min.z || p.z > b.max.z);
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
    float GetCourtNormalizedDistance()
    {
        if (!rl) return 0f;
        float cw = rl.courtHalfWidthX;
        float cz = rl.courtHalfLengthZ;

        Vector3 pos = transform.position;
        float dx = pos.x / cw;
        float dz = pos.z / cz;
        return Mathf.Clamp01(new Vector2(dx, dz).magnitude);
    }

    void RequestSmash()
    {
        // 이미 점프 중이면 또 Jump() 호출하지 않음
        if (!smashMode)
        {
            movement.Jump();
            smashMode = true;
            Debug.Log("스매시 모드 ON - 점프 시작");
        }
    }
    bool GoalIsOnMyCourt()
    {
        if (myCourtCollider == null) return false;
        GameObject goalObj = GameObject.FindGameObjectWithTag(goalTag);
        if (goalObj == null) return false;

        Bounds b = myCourtCollider.bounds;
        Vector3 g = goalObj.transform.position;

        return (g.x >= b.min.x && g.x <= b.max.x &&
                g.z >= b.min.z && g.z <= b.max.z);
    }

    Vector3 GetCourtCenter()
    {
        Bounds b = myCourtCollider.bounds;
        return new Vector3(b.center.x, transform.position.y, b.center.z);
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
            case 5: RequestSmash(); break;
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

        if (smashMode)
        {
            activeZoneType = SwingZone.ZoneType.Over;
            return false;  // 스매시 모드 중이면 다른 스윙 금지
        }
       

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
				return false;
			}
		}

		return false;
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

        prevGoalDist = 5f;
        goalDistSum = 0f;
        goalDistSteps = 0;

        swingHistory = new int[6];  // 스윙 기록 리셋

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

    public override void CollectObservations(VectorSensor sensor)
    {
        float cw = rl ? rl.courtHalfWidthX : 11f;
        float cz = rl ? rl.courtHalfLengthZ : 20f;

        // --- Player 기본 정보 ---
        Vector3 pPos = transform.position;
        bool player_courtIn = !IsOutsideCourtXZ(pPos);

        sensor.AddObservation(player_courtIn ? 1f : 0f);

        // --- 공 정보 ---
        GameObject sc = FindNearestShuttlecock();
        Vector3 ballPos = sc ? sc.transform.position : Vector3.zero;

        float ball_height = Mathf.Clamp01(ballPos.y / 10f);
        sensor.AddObservation(ball_height);

        // z방향 상대 거리
        //float ball_distance_z = Mathf.Clamp((ballPos.z - pPos.z) / cz, -1f, 1f);
        //sensor.AddObservation(ball_distance_z);

        // x방향 상대 거리
        //float ball_distance_x = Mathf.Clamp((ballPos.x - pPos.x) / cw, -1f, 1f);
        //sensor.AddObservation(ball_distance_x);

        // --- canLong / canShort ---
        float absZ = Mathf.Abs(ballPos.z);
        bool ball_canLong = absZ >= cz * 0.5f;   // 먼 거리
        bool ball_canShort = absZ < cz * 0.5f;   // 짧은 거리

        sensor.AddObservation(ball_canLong ? 1f : 0f);
        sensor.AddObservation(ball_canShort ? 1f : 0f);

        // === Over / Under Zone 기반 가능 여부 ===
        bool canOver = false;
        bool canUnder = false;

        if (overZone != null)
            canOver = overZone.GetShuttlecocks().Count > 0;

        if (underZone != null)
            canUnder = underZone.GetShuttlecocks().Count > 0;

        sensor.AddObservation(canOver ? 1f : 0f);
        sensor.AddObservation(canUnder ? 1f : 0f);

        // --- Smash 가능 여부 ---
        bool ball_canSmash = SmashDetect != null && SmashDetect.GetShuttlecocks().Count > 0;
        sensor.AddObservation(ball_canSmash ? 1f : 0f);

        // --- 공이 코트 안에 있는지 ---
        bool ball_courtIn = !IsOutsideCourtXZ(ballPos);
        sensor.AddObservation(ball_courtIn ? 1f : 0f);

        // ===== 내 턴 여부 추가 =====
        sensor.AddObservation(isMyTurn ? 1f : 0f);

        // ===== 공의 예상 착지 위치(goal) 상대 좌표 =====
        GameObject goalObj = GameObject.FindGameObjectWithTag(goalTag);
        if (goalObj != null)
        {
            Vector3 gp = goalObj.transform.position;

            float dx = gp.x - pPos.x;
            float dz = gp.z - pPos.z;

            float maxDist = Mathf.Sqrt(cw * cw + cz * cz);
            float dist = Mathf.Clamp01(Vector3.Distance(pPos, gp) / maxDist);

            //Debug.Log($"x: {dx / cw},  z: {dz / cz},  dist: {dist}");

            // 정규화된 상대 좌표 (코트 크기 기준 normalize)
            sensor.AddObservation(dx / cw);  // -1~1
            sensor.AddObservation(dz / cz);  // -1~1
            sensor.AddObservation(dist);     // 0~1
        }
        else
        {
            sensor.AddObservation(0.5f);
            sensor.AddObservation(0.5f);
            sensor.AddObservation(1f);
        }
    }

    // === 행동 적용 ===
    // 연속 2 (move XZ) + 이산 4 (오버2, 언더2)
    public override void OnActionReceived(ActionBuffers actions)
    {
        _hitGivenThisStep = false;

        var ca = actions.ContinuousActions;
        var da = actions.DiscreteActions;

        Vector2 move = new Vector2(Mathf.Clamp(ca[0], -1f, 1f), Mathf.Clamp(ca[1], -1f, 1f));
        int swing = (da.Length >= 1) ? Mathf.Clamp(da[0], 0, 5) : 0;  // 0~4 (5가지 스윙: Clear, Drop, Drive, Smash, Hairpin)
        
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

        // ==== 정답 스윙 보상 (+5) ====
        if (executedThisStep)
        {
            GameObject sc = FindNearestShuttlecock();
            if (sc != null)
            {
                Vector3 ballPos = sc.transform.position;

                float cw = rl ? rl.courtHalfWidthX : 11f;
                float cz = rl ? rl.courtHalfLengthZ : 20f;

                bool canLong = Mathf.Abs(ballPos.z) >= cz * 0.5f;
                bool canShort = !canLong;

                bool canOver = ballPos.y >= 2.0f;
                bool canUnder = !canOver;

                switch (swing)
                {
                    case 0: // Clear
                        if (canLong && canOver) AddReward(2f);
                        break;

                    case 1: // Drop
                        if (canShort && canOver) AddReward(2f);
                        break;

                    case 2: // Hairpin
                        if (canShort && canUnder) AddReward(2f);
                        break;

                    case 3: // Drive
                        if (canLong && canOver) AddReward(2f);
                        break;

                    case 4: // Under
                        if (canLong && canUnder) AddReward(2f);
                        break;

                    case 5:
                        // Smash 보상은 LateUpdate()에서 처리
                        break;
                }
            }
        }
        // ===== 목표 지점 기반 shaping reward =====
        GameObject goalObj = GameObject.FindGameObjectWithTag(goalTag);

        // === 목표 지점 상대 좌표 기반 보상 ===
        if (goalObj != null)
        {
            Vector3 gp = goalObj.transform.position;
            Vector3 pPos = transform.position;

            float cw = rl ? rl.courtHalfWidthX : 11f;
            float cz = rl ? rl.courtHalfLengthZ : 20f;

            float dxNorm = (gp.x - pPos.x) / cw;   // -1~1
            float dzNorm = (gp.z - pPos.z) / cz;   // -1~1

            float maxDist = Mathf.Sqrt(cw * cw + cz * cz);
            float distNorm = Vector3.Distance(pPos, gp) / maxDist; // 0~1

            // 0에 가까울수록 좋은 값 → 1 - |값| 으로 변환
            float xScore = 1f - Mathf.Abs(dxNorm);
            float zScore = 1f - Mathf.Abs(dzNorm);
            float distScore = 1f - distNorm;

            // 3개 평균
            float positionScore = (xScore + zScore + distScore) / 3f;

            // 가중치 적용 (0.005 정도 추천)
            float posReward = positionScore * 0.005f;

            AddReward(posReward);

            // 텐서보드 기록
            Academy.Instance.StatsRecorder.Add("eval/pos_score", positionScore);
        }

        // === 스매시 조건 체크 ===
        bool smashAvailable =
            SmashDetect != null &&
            SmashDetect.GetShuttlecocks().Count > 0;

        float courtDist = GetCourtNormalizedDistance();  // 0~1

        // 점프 시작 조건
        //if (smashAvailable && courtDist >= 0.5f)
        if (smashAvailable)
            {
            RequestSmash();
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
        else if (Input.GetKey(KeyCode.Alpha6)) da[0] = 5;  // Under
        else da[0] = 0;  // 기본값: Clear

    }




    // 셔틀콕 충돌 감지 → 보상
    void OnCollisionEnter(Collision collision)
    {
        if (_hitGivenThisStep) return;
        if (collision.gameObject.CompareTag(shuttlecockTag))
        {

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

        if (smashMode)
        {
            // 공이 SmashDetect 안에 있으면 바로 Smash
            if (overZone != null && overZone.GetShuttlecocks().Count > 0)
            {
                Debug.Log("스매시 발동!");
                shooting.Smash();
                AddReward(2f);   // ★ 스매시 보상 추가!
                smashMode = false;
            }

            // 점프가 끝났다면 스매시 모드 종료
            if (movement.isGrounded)
            {
                smashMode = false;
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
		string swingCounts = string.Format("{0}:{1}, {2}:{3}, {4}:{5}, {6}:{7}, {8}:{9}, {10}:{11}",
			SwingNames[0], _epSwingExecCounts.Length > 0 ? _epSwingExecCounts[0] : 0,
			SwingNames[1], _epSwingExecCounts.Length > 1 ? _epSwingExecCounts[1] : 0,
			SwingNames[2], _epSwingExecCounts.Length > 2 ? _epSwingExecCounts[2] : 0,
			SwingNames[3], _epSwingExecCounts.Length > 3 ? _epSwingExecCounts[3] : 0,
			SwingNames[4], _epSwingExecCounts.Length > 4 ? _epSwingExecCounts[4] : 0,
            SwingNames[5], _epSwingExecCounts.Length > 5 ? _epSwingExecCounts[5] : 0
		);

		Debug.Log(
			$"[EpLog] total={total:F3} | time={_epTimePenaltySum:F3}, center={_epCenteringSum:F3}, outCourt={_epOutOfCourtSum:F3}, wrongZone={_epWrongZoneSum:F3} (cnt={_epWrongZoneSelectCount}), swingFit={_epSwingAppropriateSum:F3}, hit={_epHitSum:F3} | swings={{ {swingCounts} }}"
		);
        float avgGoalDelta = (goalDistSteps > 0 ? goalDistSum / goalDistSteps : 0f);

        // 텐서보드 기록
        Academy.Instance.StatsRecorder.Add("eval/avg_goal_delta", avgGoalDelta);
    }
}
