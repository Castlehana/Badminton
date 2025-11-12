using UnityEngine;
using System.Collections;
using System.Diagnostics;

[RequireComponent(typeof(Rigidbody), typeof(Collider))]
public class Shuttlecock : MonoBehaviour
{
    private Rigidbody rb;
    private Collider physicsCollider;

    [Header("발사 속도 배수")]
    public float speedMultiplier = 4f;

    [Header("중력 가속도")]
    public float gravity = -50f;

    [Header("공기 저항 계수")]
    public float dragCoefficient = 0.1f;

    [Header("생존 시간 (y<=0.7 이후)")]
    public float lifeTime = 10f;

    [Header("예상 낙하 지점 표시용 프리팹 (Goal)")]
    public GameObject landingMarkerPrefab;

    private GameObject landingMarkerInstance;

    private bool lifeTimerStarted = false; // y<=0.7 이후 타이머 시작 여부
    public bool shootingLock = false;      // 발사 잠금 상태

    // 낙하 지점 영역 판정 용 변수
    private bool alreadyLanded = false;
    private bool mySide = false;
    private bool opponentSide = false;
    private bool inCourt = false;
    private bool underNet = false;

    // RallyManager 참조
    public RallyManager rallyManager;

    void Awake()
    {
        rb = GetComponent<Rigidbody>();
        physicsCollider = GetComponent<Collider>();

        rb.useGravity = false;
        rb.drag = 0;
        rb.angularDrag = 0;
        rb.collisionDetectionMode = CollisionDetectionMode.ContinuousDynamic;
        rb.interpolation = RigidbodyInterpolation.Interpolate;

        // SphereCollider만 충돌 허용   --> 이부분 수정으로 플레이어의 콜라이더를 구분할 수 있을까??(범)
        foreach (var player in GameObject.FindGameObjectsWithTag("Player"))
        {
            foreach (var col in player.GetComponentsInChildren<Collider>())
            {
                if (!(col is SphereCollider) && !col.isTrigger)
                    Physics.IgnoreCollision(physicsCollider, col, true);
            }
        }
    }

    void Start()
    {
        rallyManager = FindObjectOfType<RallyManager>();

        // 생성 직후에는 삭제 예약을 하지 않음
        alreadyLanded = false;
        mySide = false;
        opponentSide = false;
        inCourt = false;
        underNet = false;
    }

    void Update()
    {
        // y <= 0.7일 때부터 수명 타이머 시작
        if (!lifeTimerStarted && transform.position.y <= 0.7f)
        {
            StartLifeTimer();
        }
    }

    private void StartLifeTimer()
    {
        lifeTimerStarted = true;

        // 득점 판정 처리 코루틴 함수
        StartCoroutine(LandingJudgeRoutine());

        // 랠리 종료 상태로 전환
        rallyManager.State = RallyState.Ended;
        UnityEngine.Debug.Log("랠리 끝!!");

        // 셔틀콕 삭제 예약
        Destroy(gameObject, lifeTime);

        // 공유 Goal(태그 기반)도 동일 딜레이로 삭제 예약
        GameObject goalObj = GameObject.FindGameObjectWithTag("Goal");
        if (goalObj != null)
        {
            Destroy(goalObj, lifeTime);
        }

        // 개별 인스턴스를 쓴 경우에도 동일 딜레이로 삭제 예약
        if (landingMarkerInstance != null)
        {
            if (goalObj == null || landingMarkerInstance != goalObj)
                Destroy(landingMarkerInstance, lifeTime);
        }
    }

    public void Launch(float yaw, float pitch, float force)
    {
        if (shootingLock) return; // 잠금 상태면 실행 안 함

        StartCoroutine(ShootingLockRoutine());

        Quaternion rot = Quaternion.Euler(-pitch, yaw, 0f);
        Vector3 dir = rot * Vector3.forward;
        rb.velocity = dir * force * speedMultiplier;

        // 낙하 지점 예측
        Vector3 landingPos = PredictLandingPoint(yaw, pitch, force);
        //Debug.Log($"예상 낙하 지점: {landingPos}");

        // Goal 재사용 또는 생성
        GameObject goalObj = GameObject.FindGameObjectWithTag("Goal");
        if (goalObj != null)
        {
            goalObj.transform.position = landingPos;
            landingMarkerInstance = null; // 공유 Goal 사용 중
        }
        else if (landingMarkerPrefab != null)
        {
            landingMarkerInstance = Instantiate(landingMarkerPrefab, landingPos, Quaternion.identity);

            if (!landingMarkerInstance.CompareTag("Goal"))
                landingMarkerInstance.tag = "Goal";
        }
    }

    private IEnumerator ShootingLockRoutine()
    {
        shootingLock = true;
        yield return new WaitForSeconds(0.2f);
        shootingLock = false;
    }

    private Vector3 PredictLandingPoint(float yaw, float pitch, float force)
    {
        Quaternion rot = Quaternion.Euler(-pitch, yaw, 0f);
        Vector3 velocity = rot * Vector3.forward * force * speedMultiplier;
        Vector3 position = transform.position;

        float dt = Time.fixedDeltaTime;
        const float groundY = 0f;
        const int maxSteps = 10000;

        for (int i = 0; i < maxSteps; i++)
        {
            velocity.y += gravity * dt;

            Vector3 dragAccel = -dragCoefficient * velocity.magnitude * velocity;
            velocity += dragAccel * dt;

            position += velocity * dt;

            if (position.y <= groundY)
            {
                position.y = groundY;
                return position;
            }
        }
        return position;
    }

    void FixedUpdate()
    {
        float dt = Time.fixedDeltaTime;
        Vector3 v = rb.velocity;

        v.y += gravity * dt;

        Vector3 dragAccel = -dragCoefficient * v.magnitude * v;
        v += dragAccel * dt;

        rb.velocity = v;
    }

    private IEnumerator LandingJudgeRoutine()
    {
        // OnTriggerEnter 이벤트를 받을 시간 여유
        yield return new WaitForSeconds(lifeTime - 0.1f);

        // 득점 판정 처리
        if (underNet)
        {
            // 1. 네트 밑 통과 + 플레이어 코트 -> 플레이어 득점
            if (mySide) UnityEngine.Debug.Log("Player Point");
            // 2. 네트 밑 통과 + 상대 코트 -> 상대 득점
            if (opponentSide) UnityEngine.Debug.Log("AI Point");
        }
        else
        {
            if (mySide)
            {
                // 3. 플레이어 영역 + 인코트 -> 상대 득점
                if (inCourt) UnityEngine.Debug.Log("AI Point");
                // 4. 플레이어 영역 + 아웃코트 -> 플레이어 득점
                else UnityEngine.Debug.Log("Player Point");
            }
            else if (opponentSide)
            {
                // 5. 상대 영역 + 인코트 -> 플레이어 득점
                if (inCourt) UnityEngine.Debug.Log("Player Point");
                // 6. 상대 영역 + 아웃코트 -> 상대 득점
                else UnityEngine.Debug.Log("AI Point");
            }
        }
    }

    void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.CompareTag("Player")
            && collision.collider is SphereCollider)
        {
            UnityEngine.Debug.Log($"Player SphereCollider 충돌: {collision.collider.name}");
        }
    }

    void OnTriggerEnter(Collider other)
    {
        // 코트 감지
        if (other.CompareTag("MySide") && alreadyLanded == false)
        {
            alreadyLanded = true;
            mySide = true;
            //UnityEngine.Debug.Log("내쪽에 떨어짐");
        }
        if (other.CompareTag("OpponentSide") && alreadyLanded == false)
        {
            alreadyLanded = true;
            opponentSide = true;
            //UnityEngine.Debug.Log("저쪽에 떨어짐");
        }
        if (other.CompareTag("InCourt"))
        {
            inCourt = true;
            //UnityEngine.Debug.Log("코트 안에 떨어짐");
        }
        if (other.CompareTag("UnderNet"))
        {
            underNet = true;
            //UnityEngine.Debug.Log("네트 밑을 지나감");
        }
    }
}
