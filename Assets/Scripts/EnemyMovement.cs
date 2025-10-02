using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(Rigidbody), typeof(Collider))]
public class EnemyMovement : MonoBehaviour
{
    [Header("???? ????")]
    public float moveSpeed = 5f;

    [Header("???? ????")]
    public float jumpForce = 5f;

    [Header("???? ?????? (???? ??)")]
    public float gravity = -9.81f;

    [Header("?? ?????? ??????")]
    public LayerMask groundLayer;

    [Header("?? ???? Ray ???? ????")]
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

    // ????(????????)???? ???? ?????? ????
    public void SetMoveInput(Vector2 input)
    {
        moveInput = new Vector3(input.x, 0f, input.y).normalized;
    }

    // ????(????????)???? ???? ????
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

        // ?????? ???? ???? (????????): ?????? '???????? ????' ???? ?????? ????????.
        float horizontal = Input.GetAxis("Horizontal");
        float vertical = Input.GetAxis("Vertical");
        Vector2 keyboardInput = new Vector2(-horizontal, -vertical);

        if (keyboardInput.magnitude > 0.1f)
        {
            SetMoveInput(keyboardInput);
        }
        else
        {
            SetMoveInput(Vector2.zero);
        }

        // ?????? ???? ????(????????)
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
