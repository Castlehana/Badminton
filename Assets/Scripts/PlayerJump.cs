using UnityEngine;

public class PlayerJump : MonoBehaviour
{
    public float jumpForce = 7f;      // 점프 힘
    public LayerMask groundLayer;     // 땅 레이어
    public Transform groundCheck;     // 땅 체크 위치
    public float groundCheckRadius = 0.2f;

    private Rigidbody rb;
    private bool isGrounded;

    void Start()
    {
        rb = GetComponent<Rigidbody>();
    }

    void Update()
    {
        // 땅에 닿아있는지 검사
        isGrounded = Physics.CheckSphere(groundCheck.position, groundCheckRadius, groundLayer);

        // 스페이스바로 점프
        if (Input.GetKeyDown(KeyCode.Alpha9))
        {
            if (isGrounded)
            {
                UnityEngine.Debug.Log("jump!!");
                rb.AddForce(Vector3.up * jumpForce, ForceMode.Impulse);
            }
            else
                UnityEngine.Debug.Log("cant!!");
        }
    }
}
