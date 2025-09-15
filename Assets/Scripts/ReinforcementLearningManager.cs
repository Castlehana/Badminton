using UnityEngine;

public class ReinforcementLearningManager : MonoBehaviour
{
    [Header("Player Reference")]
    public Transform player; // 인스펙터에서 플레이어 Transform 할당

    [Header("셔틀콕 태그 이름")]
    public string shuttlecockTag = "Shuttlecock";

    // 코트 크기 가정(필요하면 프로젝트 값에 맞게 조정)
    [Header("Court Bounds (World, Net at Z=0)")]
    public float courtHalfWidthX = 11f;  // X: -11 ~ 11  -> -1 ~ 1
    public float courtHalfLengthZ = 20f; // Z: -20 ~ 20, 정규화는 쪽별로 -1~0 / 0~1

    void Update()
    {
        if (player == null) return;

        // Goal(예상 낙하지점) 찾기
        GameObject goalObj = GameObject.FindGameObjectWithTag("Goal");
        if (goalObj == null)
        {
            // 요구사항: Goal이 없으면 "셔틀콕 없음"으로 표시
            //Debug.Log("셔틀콕 없음");
            return;
        }

        Vector3 goalPos = goalObj.transform.position;

        // Goal의 정규화(표시용: 클램프 적용)
        float goalXNormDisp = Mathf.Clamp(goalPos.x, -courtHalfWidthX, courtHalfWidthX) / courtHalfWidthX;
        float goalZNormDisp;
        if (goalPos.z >= 0f)
            goalZNormDisp = Mathf.Clamp(goalPos.z, 0f, courtHalfLengthZ) / courtHalfLengthZ;     // 0..1
        else
            goalZNormDisp = Mathf.Clamp(goalPos.z, -courtHalfLengthZ, 0f) / courtHalfLengthZ;    // -1..0

        // 코트 밖 판정용(원본 정규화: 클램프 없이 계산)
        float goalXNormRaw = goalPos.x / courtHalfWidthX;
        float goalZNormRaw = goalPos.z / courtHalfLengthZ;
        bool isOutOfCourt = Mathf.Abs(goalXNormRaw) > 1f || Mathf.Abs(goalZNormRaw) > 1f;

        // 씬에 있는 모든 셔틀콕 검색
        GameObject[] shuttlecocks = GameObject.FindGameObjectsWithTag(shuttlecockTag);
        foreach (GameObject shuttlecock in shuttlecocks)
        {
            if (shuttlecock == null) continue;

            // 플레이어 기준 상대 위치
            Vector3 relativePos = shuttlecock.transform.position - player.position;

            // 셔틀콕의 네트(월드 원점) 기준 정규화 (표시용)
            Vector3 p = shuttlecock.transform.position;
            float xNormDisp = Mathf.Clamp(p.x, -courtHalfWidthX, courtHalfWidthX) / courtHalfWidthX;
            float zNormDisp = (p.z >= 0f)
                ? Mathf.Clamp(p.z, 0f, courtHalfLengthZ) / courtHalfLengthZ
                : Mathf.Clamp(p.z, -courtHalfLengthZ, 0f) / courtHalfLengthZ;

            // 셔틀콕 속력(m/s)
            float speed = 0f;
            Rigidbody rb = shuttlecock.GetComponent<Rigidbody>();
            if (rb != null) speed = rb.velocity.magnitude;

            //Debug.Log(
            //    $"Shuttlecock 상대 위치(플레이어 기준): X={relativePos.x:F2}, Y={relativePos.y:F2}, Z={relativePos.z:F2} | " +
            //    $"셔틀콕 정규화: X={xNormDisp:F2}, Z={zNormDisp:F2} | 속력: {speed:F2} m/s | " +
            //    $"Goal(XZ): ({goalPos.x:F2}, {goalPos.z:F2}) | " +
            //    $"Goal 정규화: X={goalXNormDisp:F2}, Z={goalZNormDisp:F2} | " +
            //    $"OUT_OF_COURT={isOutOfCourt}"
            //);
        }
    }
}
