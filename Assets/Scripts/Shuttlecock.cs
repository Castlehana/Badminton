using UnityEngine;
using System.Collections;

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

    void Awake()
    {
        rb = GetComponent<Rigidbody>();
        physicsCollider = GetComponent<Collider>();

        rb.useGravity = false;
        rb.drag = 0;
        rb.angularDrag = 0;

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
        // 생성 직후에는 삭제 예약을 하지 않음
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
        yield return new WaitForSeconds(1f);
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

    void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.CompareTag("Player")
            && collision.collider is SphereCollider)
        {
            Debug.Log($"Player SphereCollider 충돌: {collision.collider.name}");
        }
    }
}
