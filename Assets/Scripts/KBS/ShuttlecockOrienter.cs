using UnityEngine;

public class ShuttlecockOrienter : MonoBehaviour
{
    [Header("Refs")]
    public Rigidbody rb;               // 셔틀콕의 Rigidbody
    public Transform visual;           // 실제 메시(모델) Transform (없으면 자기 자신)

    [Header("Model Axis")]
    public Vector3 localHeadAxis = Vector3.forward; // 모델에서 '머리'가 향하는 로컬축(+Z면 기본값 그대로)

    [Header("Look Settings")]
    public Vector3 worldUpAxis = Vector3.up; // 코트의 위쪽(중력 반대). 보통 Vector3.up
    public float minSpeed = 0.2f;            // 너무 느릴 땐 회전 고정
    public float turnSharpness = 20f;        // 회전 응답(클수록 즉각)

    Quaternion _headAxisOffset;

    void Awake()
    {
        if (!rb) rb = GetComponent<Rigidbody>();
        if (!visual) visual = transform;

        // 모델의 '머리 방향(localHeadAxis)'을 전방(+Z)로 맵핑하는 오프셋
        _headAxisOffset = Quaternion.FromToRotation(Vector3.forward, localHeadAxis.normalized);
    }

    void FixedUpdate()
    {
        Vector3 v = rb.velocity;
        float speed = v.magnitude;
        if (speed < minSpeed) return;

        Vector3 vDir = v / speed;

        // 속도 방향을 바라보는 목표 회전 계산
        Quaternion target = Quaternion.LookRotation(vDir, worldUpAxis) * Quaternion.Inverse(_headAxisOffset);

        // 부드럽게 보간(지수 보간)
        float t = 1f - Mathf.Exp(-turnSharpness * Time.fixedDeltaTime);
        visual.rotation = Quaternion.Slerp(visual.rotation, target, t);
    }
}
