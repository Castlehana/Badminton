using UnityEngine;

public class PlayerJump : MonoBehaviour
{
    public float jumpForce = 7f;      // ���� ��
    public LayerMask groundLayer;     // �� ���̾�
    public Transform groundCheck;     // �� üũ ��ġ
    public float groundCheckRadius = 0.2f;

    private Rigidbody rb;
    private bool isGrounded;

    void Start()
    {
        rb = GetComponent<Rigidbody>();
    }

    void Update()
    {
        // ���� ����ִ��� �˻�
        isGrounded = Physics.CheckSphere(groundCheck.position, groundCheckRadius, groundLayer);

        // �����̽��ٷ� ����
        if (Input.GetKeyDown(KeyCode.Alpha9))
        {
            if (isGrounded)
            {
                UnityEngine.Debug.Log("jump!!");
                rb.AddForce(Vector3.up * jumpForce, ForceMode.Impulse);
                PlayJumpSfx();
            }
            else
                UnityEngine.Debug.Log("cant!!");
        }
    }

    private void PlayJumpSfx()
    {
        if (AudioManager.Instance == null) return;
        AudioManager.Instance.PlaySFX("Jump");
        Debug.Log("점프 소리 났슨!!!!");
    }
}
