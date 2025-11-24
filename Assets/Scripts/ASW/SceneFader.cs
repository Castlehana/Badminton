using UnityEngine;
using UnityEngine.UI;
using System.Collections;
using UnityEngine.SceneManagement;

public class SceneFader : MonoBehaviour
{
    public Image fadeImage; // 페이드 이미지
    public float fadeDuration = 1.0f; // 페이드 지속 시간

    private void Start()
    {
        // 초기 페이드 인
        if (fadeImage != null)
        {
            fadeImage.gameObject.SetActive(true); // 페이드 이미지를 활성화
            StartCoroutine(Fade(1, 0, fadeDuration, () => fadeImage.gameObject.SetActive(false))); // 투명하게
        }
    }

    // 페이드아웃 후 씬 전환
    public void FadeOutAndLoadScene(string sceneName)
    {
        if (fadeImage == null)
        {
            Debug.LogError("FadeImage가 설정되지 않았습니다.");
            return;
        }

        fadeImage.gameObject.SetActive(true); // 페이드 이미지를 활성화
        StartCoroutine(Fade(0, 1, fadeDuration, () =>
        {
            // 씬 전환 후 다시 페이드 인 실행
            SceneManager.LoadScene(sceneName);
            StartCoroutine(Fade(1, 0, fadeDuration, () => fadeImage.gameObject.SetActive(false))); // 투명하게
        }));
    }

    // 페이드 코루틴 (공통 메서드)
    private IEnumerator Fade(float startAlpha, float endAlpha, float duration, System.Action onComplete = null)
    {
        if (fadeImage == null)
        {
            yield break;
        }

        Color color = fadeImage.color;
        float elapsed = 0f;

        while (elapsed < duration)
        {
            elapsed += Time.deltaTime;
            color.a = Mathf.Lerp(startAlpha, endAlpha, elapsed / duration); // 알파 값 보간
            fadeImage.color = color;
            yield return null;
        }

        color.a = endAlpha;
        fadeImage.color = color;

        // 완료 시 실행할 콜백 호출
        onComplete?.Invoke();
    }
}
