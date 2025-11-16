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

    [Header("땅 체크 레이어")]
    public LayerMask groundLayer;

    [Header("땅 체크 Ray 길이 거리")]
    public float groundCheckDistance = 0.1f;

    private Rigidbody rb;
    private Collider col;
    private Vector3 moveInput = Vector3.zero;
    private float verticalVelocity;

    public bool trainingMode = false;

    void Awake()
    {
        rb = GetComponent<Rigidbody>();
        col = GetComponent<Collider>();
        rb.useGravity = false;
        
        // AutoMovement 컴포넌트가 있으면 비활성화 (충돌 방지)
        var autoMovement = GetComponent<AutoMovement>();
        if (autoMovement != null)
        {
            Debug.LogWarning($"[PlayerMovement] AutoMovement component found! Disabling it to prevent position override.");
            autoMovement.enabled = false;
        }
        
        // Rigidbody 제약 및 상태 확인 (디버깅)
        if (rb.constraints != RigidbodyConstraints.None && rb.constraints != RigidbodyConstraints.FreezeRotation)
        {
            Debug.LogWarning($"[PlayerMovement] Rigidbody constraints: {rb.constraints}. Movement might be restricted!");
        }
        
        if (rb.isKinematic)
        {
            Debug.LogError($"[PlayerMovement] Rigidbody is Kinematic! Movement will not work!");
        }
    }

    // 외부(에이전트)에서 이동 입력을 받음
    public void SetMoveInput(Vector2 input)
    {
        // 입력이 0이거나 매우 작으면 정규화하지 않음 (NaN 방지)
        if (input.sqrMagnitude < 0.0001f)
        {
            moveInput = Vector3.zero;
        }
        else
        {
            // 방향 벡터로 정규화 (크기는 항상 1)
            moveInput = new Vector3(input.x, 0f, input.y).normalized;
        }
        
    }

    // 외부(에이전트)에서 점프 요청
    public void Jump()
    {
        if (IsGrounded())
        {
            verticalVelocity = jumpForce;
        }
    }

    void Update()
    {
        if (trainingMode) return;

        // 키보드 이동 입력 (테스트용): 일반적으로 '에이전트 우선' 모드에서 외부 입력을 무시합니다.
        float horizontal = Input.GetAxis("Horizontal");
        float vertical = Input.GetAxis("Vertical");
        Vector2 keyboardInput = new Vector2(horizontal, vertical);

        if (keyboardInput.magnitude > 0.1f)
        {
            SetMoveInput(keyboardInput);
        }
        else
        {
            SetMoveInput(Vector2.zero);
        }

        // 키보드 점프 입력(테스트용)
        if (Input.GetKeyDown(KeyCode.Space))
        {
            Jump();
        }
    }

    void FixedUpdate()
    {
        // 학습 모드에서만 이동 처리
        if (!trainingMode) return;
        
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
