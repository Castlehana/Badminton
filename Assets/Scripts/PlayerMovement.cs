using UnityEngine;

[RequireComponent(typeof(Rigidbody), typeof(Collider))]
public class PlayerMovement : MonoBehaviour
{
    [Header("이동 속도")]
    public float moveSpeed = 5f;

    [Header("점프 힘")]
    public float jumpForce = 6f;

    [Header("땅 체크 레이어")]
    public LayerMask groundLayer;

    [Header("땅 체크 거리")]
    public float groundCheckDistance = 0.2f;

    private Rigidbody rb;
    private Collider col;
    private Vector3 moveInput = Vector3.zero;

    public bool trainingMode = false;

    public bool isGrounded;

    void Awake()
    {
        rb = GetComponent<Rigidbody>();
        col = GetComponent<Collider>();

        rb.useGravity = true;   // ★ AutoMovement와 동일한 물리 사용

        // AutoMovement 비활성화
        var autoMovement = GetComponent<AutoMovement>();
        if (autoMovement != null)
        {
            autoMovement.enabled = false;
        }
    }

    // 외부(에이전트)에서 이동 입력 전달
    public void SetMoveInput(Vector2 input)
    {
        if (input.sqrMagnitude < 0.0001f)
            moveInput = Vector3.zero;
        else
            moveInput = new Vector3(input.x, 0f, input.y).normalized;
    }

    // 외부에서 점프 요청
    public void Jump()
    {

        if (isGrounded)
        {
            rb.AddForce(Vector3.up * jumpForce, ForceMode.Impulse); // ★ 물리 점프
            Debug.Log("점프함!!!!!");
        }
        else Debug.Log("땅이아님");
    }

    void Update()
    {
        if (!trainingMode)
        {
            // 키보드 테스트 모드
            float h = Input.GetAxis("Horizontal");
            float v = Input.GetAxis("Vertical");
            SetMoveInput(new Vector2(h, v));

            if (Input.GetKeyDown(KeyCode.Space))
                Jump();
        }
    }

    void FixedUpdate()
    {
        float rayLength = col.bounds.extents.y + groundCheckDistance;

        Vector3 origin = transform.position;
        origin.y += 0.05f; // collider 내부 판정 방지

        isGrounded = Physics.Raycast(origin, Vector3.down, rayLength, groundLayer);

        Debug.DrawRay(origin, Vector3.down * rayLength, isGrounded ? Color.green : Color.red);

        if (!trainingMode) return;

        Vector3 vel = rb.velocity;
        vel.x = moveInput.x * moveSpeed;
        vel.z = moveInput.z * moveSpeed;
        rb.velocity = vel;
    }

}
