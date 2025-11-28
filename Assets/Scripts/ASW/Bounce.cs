using UnityEngine;

public class Bounce : MonoBehaviour
{
    public float amplitude = 0.2f;   // Æ¢´Â ³ôÀÌ
    public float speed = 2f;        // Æ¢´Â ¼Óµµ

    private float startY;

    void Start()
    {
        startY = transform.localPosition.y;
    }

    void Update()
    {
        float newY = startY + Mathf.Sin(Time.time * speed) * amplitude;
        transform.localPosition = new Vector3(transform.localPosition.x, newY, transform.localPosition.z);
    }
}
