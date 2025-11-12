using System.Collections;
using System.Collections.Generic;
using System.Collections.Specialized;
using UnityEngine;

public class AutoMovement : MonoBehaviour
{
    // player 이동 속도
    float moveSpeed = 10f;

    float jumpForce = 6f;

    // 충돌 컴포넌트
    Rigidbody rb;

    // RallyManager 참조
    public RallyManager rallyManager;

    [Header("Ground Check")]
    public float groundCheckDistance = 3.0f; // 얼마나 가까우면 땅으로 볼지
    public LayerMask groundLayer;            // 바닥 레이어 지정 (예: Ground)

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

        // 점프 입력 처리 (Y는 오직 여기서만 변함)
        if (Input.GetKeyDown(KeyCode.Alpha7) && isGrounded && rallyManager.State == RallyState.Rallying)
        {
            Jump();
        }

    }

    public void Jump()
    {
        UnityEngine.Debug.Log("점프!!");
        rb.AddForce(Vector3.up * jumpForce, ForceMode.Impulse);
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

        // 현재 내 위치와 골 위치, 중심 구하기
        Vector3 myPos;
        Vector3 destPos;
        Vector3 centerPos;

        myPos = rb.position;
        destPos = myPos;

        if (goalObj != null)
        {
            destPos = goalObj.transform.position;
            destPos.y = myPos.y;
        }
        centerPos.x = 0.0f;
        centerPos.y = myPos.y;
        centerPos.z = 10.0f;

        // 내 코트쪽에 'Goal' 태그가 있으면 따라가기
        if (goalObj != null && destPos.z > 0)
        {
            Vector3 nextPos = Vector3.MoveTowards(myPos, destPos, moveSpeed * Time.fixedDeltaTime);
            rb.MovePosition(nextPos);
        }
        // 골 목표가 사라지면 다시 중심으로 복귀
        else
        {
            Vector3 nextPos = Vector3.MoveTowards(myPos, centerPos, moveSpeed * Time.fixedDeltaTime);
            rb.MovePosition(nextPos);
        }
    }
}