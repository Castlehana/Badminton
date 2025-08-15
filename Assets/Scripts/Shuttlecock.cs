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

    [Header("예상 낙하 지점 표시용 프리팹 (Goal)")]
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

        // 낙하 마커도 함께 제거 (개별 인스턴스를 썼을 때만)
        if (landingMarkerInstance != null)
            Destroy(landingMarkerInstance, lifeTime);
    }

    public void Launch(float yaw, float pitch, float force)
    {
        Quaternion rot = Quaternion.Euler(-pitch, yaw, 0f);
        Vector3 dir = rot * Vector3.forward;
        rb.velocity = dir * force * speedMultiplier;

        // 낙하 지점 예측
        Vector3 landingPos = PredictLandingPoint(yaw, pitch, force);
        Debug.Log($"예상 낙하 지점: {landingPos}");

        // Goal 재사용 또는 생성
        GameObject goalObj = GameObject.FindGameObjectWithTag("Goal");
        if (goalObj != null)
        {
            // 이미 있으면 위치만 갱신
            goalObj.transform.position = landingPos;
            landingMarkerInstance = null; // 공유 Goal을 쓰는 경우 개별 파괴 예약 안 함
        }
        else if (landingMarkerPrefab != null)
        {
            // 없으면 새로 생성
            landingMarkerInstance = Instantiate(landingMarkerPrefab, landingPos, Quaternion.identity);

            // 프리팹 태그 미설정 대비
            if (!landingMarkerInstance.CompareTag("Goal"))
                landingMarkerInstance.tag = "Goal";
        }
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

    // 셔틀콕이 삭제될 때 Goal 오브젝트도 삭제 (공유 Goal을 계속 유지하고 싶다면 이 메서드를 제거하세요)
    void OnDestroy()
    {
        GameObject goalObj = GameObject.FindGameObjectWithTag("Goal");
        if (goalObj != null)
        {
            Destroy(goalObj);
        }
    }
}
