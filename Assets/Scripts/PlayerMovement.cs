//using System.Diagnostics;
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
    private Vector3 moveInput = Vector3.zero;
    private float verticalVelocity;

    void Awake()
    {
        rb = GetComponent<Rigidbody>();
        col = GetComponent<Collider>();
        rb.useGravity = false;
    }

    // 외부(아두이노)에서 이동 입력을 설정
    public void SetMoveInput(Vector2 input)
    {
        moveInput = new Vector3(input.x, 0f, input.y).normalized;
    }

    // 외부(아두이노)에서 점프 요청
    public void Jump()
    {
        if (IsGrounded())
        {
            verticalVelocity = jumpForce;
        }
    }

    void Update()
    {
        // 키보드로도 점프 가능 (선택 사항)
        if (Input.GetKeyDown(KeyCode.Space))
        {
            Jump();
        }
    }

    void FixedUpdate()
    {
        float dt = Time.fixedDeltaTime;
        bool grounded = IsGrounded();

        if (grounded && verticalVelocity < 0f)
        {
            verticalVelocity = 0f;
        }

        verticalVelocity += gravity * dt;

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
