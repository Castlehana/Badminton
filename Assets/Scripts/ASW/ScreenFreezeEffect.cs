using System.Collections;
using UnityEngine;
using UnityEngine.UI;

public class ScreenFreezeEffect : MonoBehaviour
{
    [Header("얼린 화면을 보여줄 RawImage (전체 화면)")]
    public RawImage freezeImage;

    [Header("얼려두는 시간 (초)")]
    public float holdDuration = 0.5f;

    [Header("페이드 아웃 시간 (초)")]
    public float fadeDuration = 0.2f;

    private bool _isRunning = false;

    void Awake()
    {
        if (freezeImage != null)
        {
            // 화면 전체를 덮도록 RectTransform 강제
            RectTransform rt = freezeImage.rectTransform;
            rt.anchorMin = Vector2.zero;
            rt.anchorMax = Vector2.one;
            rt.offsetMin = Vector2.zero;
            rt.offsetMax = Vector2.zero;

            // 처음에는 안 보이게
            Color c = freezeImage.color;
            c.a = 0f;
            freezeImage.color = c;
            freezeImage.gameObject.SetActive(false);

            // 텍스처 비율 보정 끄기 (불필요한 스케일링 방지)
            //freezeImage.preserveAspect = false;
        }
    }

    // RallyManager 등에서 호출: screenFreeze.PlayFreeze();
    public void PlayFreeze()
    {
        if (!_isRunning && freezeImage != null)
        {
            StartCoroutine(FreezeRoutine());
        }
    }

    private IEnumerator FreezeRoutine()
    {
        _isRunning = true;

        // 프레임 렌더링이 끝난 뒤 캡처
        yield return new WaitForEndOfFrame();

        int w = Screen.width;
        int h = Screen.height;

        // RenderTexture로 화면 캡처
        RenderTexture rt = RenderTexture.GetTemporary(w, h, 24, RenderTextureFormat.ARGB32);
        ScreenCapture.CaptureScreenshotIntoRenderTexture(rt);

        RenderTexture prev = RenderTexture.active;
        RenderTexture.active = rt;

        Texture2D screenTex = new Texture2D(w, h, TextureFormat.RGBA32, false);
        screenTex.ReadPixels(new Rect(0, 0, w, h), 0, 0);
        screenTex.Apply();

        RenderTexture.active = prev;
        RenderTexture.ReleaseTemporary(rt);

        // 상하 반전 보정
        FlipTextureVertically(screenTex);

        // RawImage에 텍스처 적용
        freezeImage.texture = screenTex;
        freezeImage.gameObject.SetActive(true);

        // 색은 항상 (1,1,1,alpha): 원본 색 그대로, 투명도만 조절
        freezeImage.color = new Color(1f, 1f, 1f, 1f);

        // 얼린 화면 유지
        yield return new WaitForSecondsRealtime(holdDuration);

        // 페이드 아웃
        float t = 0f;
        while (t < fadeDuration)
        {
            t += Time.unscaledDeltaTime;
            float alpha = 1f - Mathf.Clamp01(t / fadeDuration);
            freezeImage.color = new Color(1f, 1f, 1f, alpha);
            yield return null;
        }

        // 정리
        freezeImage.color = new Color(1f, 1f, 1f, 0f);
        freezeImage.gameObject.SetActive(false);

        if (screenTex != null)
        {
            Destroy(screenTex);
        }

        _isRunning = false;
    }

    // 상하 반전 보정 함수
    private void FlipTextureVertically(Texture2D tex)
    {
        int w = tex.width;
        int h = tex.height;

        Color[] src = tex.GetPixels();
        Color[] dst = new Color[src.Length];

        for (int y = 0; y < h; y++)
        {
            int flippedY = h - 1 - y;
            for (int x = 0; x < w; x++)
            {
                dst[flippedY * w + x] = src[y * w + x];
            }
        }

        tex.SetPixels(dst);
        tex.Apply();
    }
}
