using UnityEngine;

public class ShuttlecockCameraController : MonoBehaviour
{
    [Header("셔틀콕 탐색 설정")]
    public string shuttleTag = "Shuttlecock"; // 셔틀콕 태그

    [Header("카메라 회전 설정 (셔틀 Y에 따라 X 회전 변경)")]
    public float baseRotationX = 30f;   // 셔틀 Y <= 15일 때 카메라 기본 X 회전값
    public float thresholdY = 15f;      // 기준이 되는 셔틀콕의 Y값
    public float rotationPerUnitY = 1f; // Y가 1 오를 때 X를 몇도 줄일지

    [Header("부드러운 따라가기 설정")]
    public Transform followTarget;              // 인스펙터에서 할당할 대상(플레이어 등)
    public Vector3 followOffset = new Vector3(0f, 5f, -10f); // 타겟 기준 위치 오프셋(기본)
    public float followSmoothTime = 0.2f;       // 값이 클수록 더 느리게 따라감

    [Header("셔틀 Y에 따른 Z 변화")]
    public float zPerUnitY = -0.2f;             // 셔틀 Y가 1 증가할 때 offset.z에 더해질 값

    [Header("포지션 X에 따른 Y 회전")]
    public float basePosXForY = 0f;             // 포지션 X 기준값 (0일 때)
    public float baseRotationY = 180f;          // 포지션 X == 0일 때 Y 회전 기본값
    public float yPerUnitPosX = 0.5f;           // 포지션 X가 1 증가할 때 Y를 얼마나 증가시킬지

    private GameObject shuttle;
    private Vector3 followVelocity = Vector3.zero;

    void Start()
    {
        FindShuttle();
    }

    void LateUpdate()
    {
        float deltaY = 0f;

        // 셔틀 Y에 따른 deltaY 계산
        if (shuttle == null)
        {
            FindShuttle();
        }

        if (shuttle != null)
        {
            float shuttleY = shuttle.transform.position.y;
            deltaY = shuttleY - thresholdY;

            // 기준보다 아래로 내려가면 deltaY는 0 (기본 상태 유지)
            if (deltaY < 0f)
                deltaY = 0f;
        }

        // 셔틀 Y가 15를 넘으면, 1마다 Z를 -0.2씩 (기본 followOffset.z에서 더 멀어짐)
        Vector3 dynamicOffset = followOffset;
        dynamicOffset.z += deltaY * zPerUnitY;

        // 1) 타겟을 부드럽게 따라가기 (동적으로 바뀐 offset 사용)
        if (followTarget != null)
        {
            Vector3 targetPos = followTarget.position + dynamicOffset;
            transform.position = Vector3.SmoothDamp(
                transform.position,
                targetPos,
                ref followVelocity,
                followSmoothTime
            );
        }

        // 2) 회전 X: 셔틀 Y에 따라 카메라 X 회전 조정
        float targetX = baseRotationX - deltaY * rotationPerUnitY;

        // 3) 회전 Y: "카메라 포지션 X"에 비례해서 변경
        //    - posX = 0  → Y = 180
        //    - posX = +1 → Y = 180 + 0.5
        //    - posX = -1 → Y = 180 - 0.5
        float posX = transform.position.x;
        float deltaPosX = posX - basePosXForY;
        float targetY = baseRotationY + deltaPosX * yPerUnitPosX;

        // 4) 회전 적용 (X, Y만 갱신)
        Vector3 euler = transform.eulerAngles;
        euler.x = targetX;
        euler.y = targetY;
        transform.eulerAngles = euler;
    }

    void FindShuttle()
    {
        shuttle = GameObject.FindWithTag(shuttleTag);
    }
}
