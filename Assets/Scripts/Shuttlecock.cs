using UnityEngine;

[RequireComponent(typeof(Rigidbody))]
public class Shuttlecock : MonoBehaviour
{
    private Rigidbody rb;
    private bool isSpeedFixed = false;
    private Coroutine speedFixCoroutine; // 현재 실행 중인 감속 코루틴

    void Awake()
    {
        rb = GetComponent<Rigidbody>();
    }

    void Start()
    {
        Destroy(gameObject, 10f); // 10초 후 자동 삭제
    }

    public void Launch(float yawAngle, float pitchAngle, float launchForce)
    {
        // 이전 속도 고정 상태 초기화
        isSpeedFixed = false;

        // 기존 감속 코루틴이 돌고 있었다면 멈춤
        if (speedFixCoroutine != null)
        {
            StopCoroutine(speedFixCoroutine);
        }

        // 방향 계산
        Quaternion yawRotation = Quaternion.Euler(0f, yawAngle, 0f);
        Quaternion fullRotation = yawRotation * Quaternion.Euler(-pitchAngle, 0f, 0f);
        Vector3 launchDir = fullRotation * Vector3.forward;

        // 기존 속도 초기화
        rb.velocity = Vector3.zero;
        rb.angularVelocity = Vector3.zero;

        // 초기 속도 적용
        Vector3 initialVelocity = launchDir * launchForce;
        rb.velocity = initialVelocity;

        // 감속 시작
        speedFixCoroutine = StartCoroutine(AdjustSpeedToFixed(initialVelocity.normalized, 10f, 0.5f));
    }

    private System.Collections.IEnumerator AdjustSpeedToFixed(Vector3 direction, float targetSpeed, float duration)
    {
        float time = 0f;
        float startSpeed = rb.velocity.magnitude;

        while (time < duration)
        {
            time += Time.deltaTime;
            float t = time / duration;

            // 선형 보간으로 감속
            float currentSpeed = Mathf.Lerp(startSpeed, targetSpeed, t);
            rb.velocity = direction * currentSpeed;

            yield return null;
        }

        // 마지막 속도 고정
        rb.velocity = direction * targetSpeed;
        isSpeedFixed = true;
    }

    void FixedUpdate()
    {
        // 속도 고정 상태면 계속 유지
        if (isSpeedFixed)
        {
            Vector3 dir = rb.velocity.normalized;
            rb.velocity = dir * 5f;
        }
    }
}
