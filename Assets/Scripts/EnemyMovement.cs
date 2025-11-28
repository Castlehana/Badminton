using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(Rigidbody), typeof(Collider))]
public class EnemyMovement : MonoBehaviour
{
    [Header("�̵� �ӵ�")]
    public float moveSpeed = 5f;

    [Header("���� �ӵ�")]
    public float jumpForce = 5f;

    [Header("�߷� ���ӵ� (���� ��)")]
    public float gravity = -9.81f;

    [Header("�� üũ�� ���̾�")]
    public LayerMask groundLayer;

    [Header("�� üũ Ray �߰� ����")]
    public float groundCheckDistance = 0.1f;

    private Rigidbody rb;
    private Collider col;
    private Vector3 moveInput = Vector3.zero;
    private float verticalVelocity;

    public bool trainingMode = true;

    void Awake()
    {
        rb = GetComponent<Rigidbody>();
        col = GetComponent<Collider>();
        rb.useGravity = false;
    }

    // �ܺ�(�Ƶ��̳�)���� �̵� �Է��� ����
    public void SetMoveInput(Vector2 input)
    {
        moveInput = new Vector3(input.x, 0f, input.y).normalized;
    }

    // �ܺ�(�Ƶ��̳�)���� ���� ��û
    public void Jump()
    {
        if (IsGrounded())
        {
            verticalVelocity = jumpForce;
            PlayJumpSfx();
        }
    }

    void Update()
    {
        if (trainingMode)
        {
            //Debug.Log(trainingMode);
            return;
        }

        // Ű���� �̵� �Է� (�׽�Ʈ��): ������ '�켱���� ����' �ܺ� �Է��� �����.
        float horizontal = Input.GetAxis("Horizontal");
        float vertical = Input.GetAxis("Vertical");
        Vector2 keyboardInput = new Vector2(-horizontal, -vertical);

        if (keyboardInput.magnitude > 0.01f)
        {
            SetMoveInput(keyboardInput);
            
        }
        else
        {
            SetMoveInput(Vector2.zero); 
        }

        // Ű���� ���� �Է�(�׽�Ʈ��)
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

    private void PlayJumpSfx()
    {
        if (AudioManager.Instance == null) return;
        AudioManager.Instance.PlaySFX("Jump");
    }
}
