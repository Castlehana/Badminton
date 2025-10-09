using System.Collections;
using System.Collections.Generic;
using System.Collections.Specialized;
using UnityEngine;

public class AutoMovement : MonoBehaviour
{
    // player 이동 속도
    float moveSpeed = 10f;

    // 충돌 컴포넌트
    Rigidbody rb;

    // RallyManager 참조
    public RallyManager rallyManager;

    // Start is called before the first frame update
    void Start()
    {
        rb = GetComponent<Rigidbody>();
        rb.useGravity = true;
        rb.constraints = RigidbodyConstraints.FreezePositionX | RigidbodyConstraints.FreezePositionZ;
        rb.constraints = RigidbodyConstraints.FreezeRotationX | RigidbodyConstraints.FreezeRotationY | RigidbodyConstraints.FreezeRotationZ;
    }

    // Update is called once per frame
    void Update()
    {
        if (rallyManager != null && rallyManager.State == RallyState.Ready)
        {
            transform.position = new Vector3(0f, 3f, 10f);
            return;
        }

        GameObject goalObj = GameObject.FindGameObjectWithTag("Goal");

        // 현재 내 위치와 골 위치, 중심 구하기
        Vector3 myPos;
        Vector3 destPos;
        Vector3 centerPos;

        myPos = transform.position;
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
            transform.position = Vector3.MoveTowards(myPos, destPos, moveSpeed * Time.deltaTime);
        }
        // 골 목표가 사라지면 다시 중심으로 복귀
        else
        {
            transform.position = Vector3.MoveTowards(myPos, centerPos, moveSpeed * Time.deltaTime);
        }

    }

    // Goal을 추적해 이동하는 함수
    private void FollowGoal(GameObject goalObj)
    {
        // 현재 내 위치와 골 위치를 먼저 받아오기
        Vector3 myPos;
        Vector3 destPos;

        myPos = transform.position;
        destPos = goalObj.transform.position;
        destPos.y = myPos.y;

        // 골 위치가 내 코트 쪽인지 확인하기
        if (destPos.z > 0)
        {
            transform.position = Vector3.MoveTowards(myPos, destPos, moveSpeed * Time.deltaTime);
            //dir.x = destPos.x - myPos.x;
            //dir.z = destPos.z - myPos.z;
            //dir.y = 0;

            //transform.Translate(dir.normalized * Time.deltaTime);
        }
    }

    // 원래 위치 (0, -10)으로 돌아오는 함수
    private void ReturnPos()
    {
        // 현재 내 위치와 중심 위치를 먼저 받아오기
        Vector3 myPos;
        Vector3 centerPos;

        myPos = transform.position;
        centerPos.x = 0.0f;
        centerPos.y = myPos.y;
        centerPos.z = 10.0f;

        transform.position = Vector3.MoveTowards(myPos, centerPos, moveSpeed * Time.deltaTime);
    }
}