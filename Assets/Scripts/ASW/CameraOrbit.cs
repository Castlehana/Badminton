using UnityEngine;

public class CameraOrbit : MonoBehaviour
{
    public float rotationSpeed = 5f; // 회전 속도

    void Update()
    {
        // Y축을 기준으로 회전
        transform.Rotate(0, rotationSpeed * Time.deltaTime, 0);
    }
}
