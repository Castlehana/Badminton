using System.Collections;
using System.Collections.Generic;
using System.Collections.Specialized;
using UnityEngine;

public class AutoMovement : MonoBehaviour
{
    // player �̵� �ӵ�
    float moveSpeed = 10f;

    public float jumpForce = 9f;

    // �浹 ������Ʈ
    Rigidbody rb;

    // RallyManager ����
    public RallyManager rallyManager;

    [Header("Ground Check")]
    public float groundCheckDistance = 3.0f; // �󸶳� ������ ������ ����
    public LayerMask groundLayer;            // �ٴ� ���̾� ���� (��: Ground)

    public bool isGrounded;

    public bool isJumping => !isGrounded;

    // Start is called before the first frame update
    void Start()
    {
        rb = GetComponent<Rigidbody>();
        rb.useGravity = true;
        rb.constraints = RigidbodyConstraints.FreezeRotationX | RigidbodyConstraints.FreezeRotationY | RigidbodyConstraints.FreezeRotationZ;
    }

    // Update is called once per frame
    void Update()
    {
        isGrounded = Physics.Raycast(transform.position, Vector3.down, groundCheckDistance + 0.1f, groundLayer);

        // ���� �Է� ó�� (Y�� ���� ���⼭�� ����)
        if (Input.GetKeyDown(KeyCode.Alpha7))
        {
            Jump();
        }

    }

    public void Jump()
    {
        if (isGrounded && rallyManager.State == RallyState.Rallying)
        {
            UnityEngine.Debug.Log("짬푸짬푸!!");
            rb.AddForce(Vector3.up * jumpForce, ForceMode.Impulse);
            PlayJumpSfx();
        }
    }

    void FixedUpdate()
    {
        if (rallyManager != null && rallyManager.State == RallyState.Ready)
        {
            if(rallyManager.Mode != ModeState.Training)
            {
                rb.MovePosition(new Vector3(0f, 3f, 10f));
                return;
            }
        }

        GameObject goalObj = GameObject.FindGameObjectWithTag("Goal");

        // ���� �� ��ġ�� �� ��ġ, �߽� ���ϱ�
        Vector3 myPos;
        Vector3 destPos;
        Vector3 centerPos;

        myPos = rb.position;
        destPos = myPos;

        if (goalObj != null)
        {
            destPos = goalObj.transform.position;
            destPos.y = myPos.y;
            //destPos.z += 1.5f;
        }
        centerPos.x = 0.0f;
        centerPos.y = myPos.y;
        centerPos.z = 10.0f;

        // �� ��Ʈ�ʿ� 'Goal' �±װ� ������ ���󰡱�
        if (goalObj != null && destPos.z > 0)
        {
            Vector3 nextPos = Vector3.MoveTowards(myPos, destPos, moveSpeed * Time.fixedDeltaTime);
            rb.MovePosition(nextPos);
        }
        // �� ��ǥ�� ������� �ٽ� �߽����� ����
        else
        {
            Vector3 nextPos = Vector3.MoveTowards(myPos, centerPos, moveSpeed * Time.fixedDeltaTime);
            rb.MovePosition(nextPos);
        }
    }

    private void PlayJumpSfx()
    {
        if (AudioManager.Instance == null) return;
        AudioManager.Instance.PlaySFX("Jump");
    }
}