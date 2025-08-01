//using System.Diagnostics;
using UnityEngine;

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

    [Header("생존 시간")]
    public float lifeTime = 10f;

    [Header("예상 낙하 지점 표시용 프리팹")]
    public GameObject landingMarkerPrefab;

    private GameObject landingMarkerInstance;

    void Awake()
    {
        rb = GetComponent<Rigidbody>();
        physicsCollider = GetComponent<Collider>();

        rb.useGravity = false;
        rb.drag = 0;
        rb.angularDrag = 0;

        // SphereCollider만 충돌 허용
        foreach (var player in GameObject.FindGameObjectsWithTag("Player"))
        {
            foreach (var col in player.GetComponentsInChildren<Collider>())
            {
                if (!(col is SphereCollider))
                    Physics.IgnoreCollision(physicsCollider, col, true);
            }
        }
    }

    void Start()
    {
        // 셔틀콕 제거 예약
        Destroy(gameObject, lifeTime);

        // 낙하 마커도 함께 제거
        if (landingMarkerInstance != null)
            Destroy(landingMarkerInstance, lifeTime);
    }

    /// <summary>
    /// 지정한 방향과 힘으로 셔틀콕 발사
    /// </summary>
    public void Launch(float yaw, float pitch, float force)
    {
        Quaternion rot = Quaternion.Euler(-pitch, yaw, 0f);
        Vector3 dir = rot * Vector3.forward;
        rb.velocity = dir * force * speedMultiplier;

        // 낙하 지점 예측
        Vector3 landingPos = PredictLandingPoint(yaw, pitch, force);
        Debug.Log($"예상 낙하 지점: {landingPos}");

        // 마커 생성
        if (landingMarkerPrefab != null)
        {
            landingMarkerInstance = Instantiate(landingMarkerPrefab, landingPos, Quaternion.identity);
        }
    }

    /// <summary>
    /// 공중 궤적을 시뮬레이션하여 y=0 도달 위치 예측
    /// </summary>
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
            // 중력 적용
            velocity.y += gravity * dt;

            // 항력 적용
            Vector3 dragAccel = -dragCoefficient * velocity.magnitude * velocity;
            velocity += dragAccel * dt;

            // 위치 갱신
            position += velocity * dt;

            // y=0 이하로 내려오면 종료
            if (position.y <= groundY)
            {
                position.y = groundY;
                return position;
            }
        }

        // 너무 오래 걸릴 경우 마지막 위치 반환
        return position;
    }

    void FixedUpdate()
    {
        float dt = Time.fixedDeltaTime;
        Vector3 v = rb.velocity;

        // 중력 적용
        v.y += gravity * dt;

        // 항력 적용
        Vector3 dragAccel = -dragCoefficient * v.magnitude * v;
        v += dragAccel * dt;

        // 결과 속도 반영
        rb.velocity = v;
    }

    void OnCollisionEnter(Collision collision)
    {
        // Player의 SphereCollider에만 반응
        if (collision.gameObject.CompareTag("Player")
            && collision.collider is SphereCollider)
        {
            Debug.Log($"Player SphereCollider 충돌: {collision.collider.name}");
            // 충돌 처리 로직 추가 가능
        }
    }
}
