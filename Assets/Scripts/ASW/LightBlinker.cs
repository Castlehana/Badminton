using UnityEngine;
using System.Collections;

public class LightBlinker : MonoBehaviour
{
    public Light targetLight;
    public float intensityOn = 3f;      // ±âº» ¹à±â
    public float intensityOff = 0f;     // ²¨Áú ¶§ ¹à±â
    public float holdDuration = 3f;     // ±ôºý ½ÃÄö½º ÀüÈÄ À¯Áö ½Ã°£
    public float blinkInterval = 0.1f;  // ²¨Á³´Ù ÄÑÁö´Â ½Ã°£ °£°Ý
    public int blinkCount = 3;          // ±ôºý È½¼ö

    void Start()
    {
        if (targetLight == null)
            targetLight = GetComponent<Light>();

        targetLight.intensity = intensityOn;
        StartCoroutine(BlinkLoop());
    }

    IEnumerator BlinkLoop()
    {
        while (true)
        {
            // 3ÃÊ À¯Áö
            yield return new WaitForSeconds(holdDuration);

            // ±ôºý 3¹ø
            for (int i = 0; i < blinkCount; i++)
            {
                targetLight.intensity = intensityOff;  // ²¨Áü
                yield return new WaitForSeconds(blinkInterval);

                targetLight.intensity = intensityOn;   // ÄÑÁü
                yield return new WaitForSeconds(blinkInterval);
            }
        }
    }
}
