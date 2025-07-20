
using UnityEngine;

[RequireComponent(typeof(Rigidbody), typeof(Collider))]
public class PlayerMovement : MonoBehaviour
{
    [Header("이동 속도")]
    public float moveSpeed = 5f;

    [Header("점프 속도")]
    public float jumpForce = 5f;

    [Header("중력 가속도 (음수 값)")]
    public float gravity = -9.81f;

    [Header("땅 체크용 레이어")]
    public LayerMask groundLayer;

    [Header("땅 체크 Ray 추가 길이")]
    public float groundCheckDistance = 0.1f;

    private Rigidbody rb;
    private Collider col;
    private Vector3 moveInput;
    private float verticalVelocity;  // 수동 중력을 위한 Y속도

    void Awake()
    {
        rb = GetComponent<Rigidbody>();
        col = GetComponent<Collider>();
        rb.useGravity = false;   // 내장 중력 끔
    }

    void Update()
    {
        // 이동 입력
        float h = Input.GetAxisRaw("Horizontal");
        float v = Input.GetAxisRaw("Vertical");
        moveInput = new Vector3(h, 0f, v).normalized;

        // 점프 입력
        if (Input.GetKeyDown(KeyCode.Space) && IsGrounded())
        {
            verticalVelocity = jumpForce;  // 수동으로 Y속도 설정
        }
    }

    void FixedUpdate()
    {
        float dt = Time.fixedDeltaTime;
        bool grounded = IsGrounded();

        // 땅에 닿아 있고 아래로 향하는 속도면 Y속도 리셋
        if (grounded && verticalVelocity < 0f)
        {
            verticalVelocity = 0f;
        }

        // 수동 중력 적용
        verticalVelocity += gravity * dt;

        // 최종 속도 계산
        Vector3 velocity = new Vector3(
            moveInput.x * moveSpeed,
            verticalVelocity,
            moveInput.z * moveSpeed
        );

        rb.velocity = velocity;
    }

    bool IsGrounded()
    {
        Vector3 origin = transform.position;
        float rayLength = col.bounds.extents.y + groundCheckDistance;

        Debug.DrawRay(origin, Vector3.down * rayLength, Color.red);
        return Physics.Raycast(origin, Vector3.down, rayLength, groundLayer);
    }
}
