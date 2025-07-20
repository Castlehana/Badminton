
using UnityEngine;

[RequireComponent(typeof(Rigidbody), typeof(Collider))]
public class Shuttlecock : MonoBehaviour
{
    private Rigidbody rb;
    private Collider physicsCollider;

    [Header("발사 속도 배수")]
    public float speedMultiplier = 4f;

    [Header("중력 가속도")]
    public float gravity = -50f;    // m/s²

    [Header("공기 저항 계수")]
    public float dragCoefficient = 0.1f; // 클수록 빠르게 속도 감소

    [Header("생존 시간")]
    public float lifeTime = 10f;    // 발사 후 사라지기까지 시간

    void Awake()
    {
        rb = GetComponent<Rigidbody>();
        physicsCollider = GetComponent<Collider>();

        // 유니티 내장 중력 대신 수동 적용
        rb.useGravity = false;
        rb.drag = 0;
        rb.angularDrag = 0;

        // 플레이어 태그 가진 오브젝트들의 Collider 중
        // SphereCollider만 충돌 허용, 나머지는 무시
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
        Destroy(gameObject, lifeTime);
    }

    /// <summary>
    /// 지정한 방향과 힘으로 셔틀콕 발사
    /// </summary>
    public void Launch(float yaw, float pitch, float force)
    {
        Quaternion rot = Quaternion.Euler(-pitch, yaw, 0f);
        Vector3 dir = rot * Vector3.forward;
        rb.velocity = dir * force * speedMultiplier;
    }

    void FixedUpdate()
    {
        float dt = Time.fixedDeltaTime;
        Vector3 v = rb.velocity;

        // 1) 중력 가속도 적용
        v.y += gravity * dt;

        // 2) 공기 저항 (Quadratic drag) 적용
        Vector3 dragAccel = -dragCoefficient * v.magnitude * v;
        v += dragAccel * dt;

        rb.velocity = v;
    }

    void OnCollisionEnter(Collision collision)
    {
        // 오직 플레이어의 SphereCollider와 충돌했을 때만 인식
        if (collision.gameObject.CompareTag("Player")
            && collision.collider is SphereCollider)
        {
            Debug.Log($"Player SphereCollider 충돌: {collision.collider.name}");
            // 여기에 충돌 처리 로직 추가
        }
    }
}
