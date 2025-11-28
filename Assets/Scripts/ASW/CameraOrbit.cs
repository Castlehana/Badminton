using UnityEngine;

public class CameraOrbit : MonoBehaviour
{
    public float rotationSpeed = 30f; // 초당 회전 속도(도)
    public float switchInterval = 10f; // 방향 바꾸는 간격

    private float timer = 0f;
    private int direction = 1; // 1 = 시계방향, -1 = 반시계방향

    void Update()
    {
        timer += Time.deltaTime;

        // 일정 시간마다 방향 반전
        if (timer >= switchInterval)
        {
            direction *= -1;   // 방향 뒤집기
            timer = 0f;        // 타이머 초기화
        }

        // 회전 적용
        transform.Rotate(0, direction * rotationSpeed * Time.deltaTime, 0);
    }
}
