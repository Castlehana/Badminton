using UnityEngine;

public class Shooter : MonoBehaviour
{
    public GameObject shuttlecockPrefab;   // Inspector에서 프리팹 지정

    [Header("XZ 방향 각도 범위 (Y축 회전)")]
    public float minYaw = 170f;
    public float maxYaw = 190f;

    [Header("위쪽 각도 범위 (Pitch)")]
    public float minPitch = 30f;
    public float maxPitch = 60f;

    [Header("발사 속도 범위")]
    public float minForce = 30f;
    public float maxForce = 60f;

    [Header("발사 주기 (초)")]
    public float fireInterval = 5f;

    private float timer = 0f;

    void Update()
    {
        timer += Time.deltaTime;
        if (timer >= fireInterval)
        {
            timer = 0f;
            FireRandomShuttlecock();
        }
    }

    void FireRandomShuttlecock()
    {
        // UnityEngine.Random 명시적으로 사용
        float yaw = UnityEngine.Random.Range(minYaw, maxYaw);
        float pitch = UnityEngine.Random.Range(minPitch, maxPitch);
        float force = UnityEngine.Random.Range(minForce, maxForce);

        // 현재 위치에서 셔틀콕 생성
        GameObject shuttle = Instantiate(shuttlecockPrefab, transform.position, Quaternion.identity);

        // 포물선 발사
        shuttle.GetComponent<Shuttlecock>().Launch(yaw, pitch, force);
    }
}
