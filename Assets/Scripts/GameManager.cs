using System.Collections;
using UnityEngine;
using UnityEngine.Events;

public class GameManager : MonoBehaviour
{
    [Header("배속 범위")]
    [Range(0.01f, 1f)] public float minTimeScale = 0.1f;
    [Range(1f, 20f)] public float maxTimeScale = 10f;

    [Header("초기 배속")]
    [Range(0.1f, 10f)] public float initialTimeScale = 1f;

    [Header("옵션")]
    public bool applyOnStart = true;
    public bool showOnGUISlider = true;     // 화면 좌상단 슬라이더 표시
    public bool enableHotkeys = true;       // 단축키로 조절

    [Header("이벤트")]
    public UnityEvent<float> onTimeScaleChanged; // 변경된 timeScale 전달

    private float baseFixedDeltaTime;
    private Coroutine lerpRoutine;
    private float _targetScale; // 실시간 제어용

    public bool IsPaused => Time.timeScale <= Mathf.Epsilon;
    public float CurrentTimeScale => Time.timeScale;

    void Awake()
    {
        baseFixedDeltaTime = Time.fixedDeltaTime;
        _targetScale = Mathf.Clamp(initialTimeScale, minTimeScale, maxTimeScale);
    }

    void Start()
    {
        if (applyOnStart) ApplyTimeScale(_targetScale);
    }

    void Update()
    {
        if (!enableHotkeys) return;

        // 단축키: [ 키로 감소, ] 키로 증가
        if (Input.GetKeyDown(KeyCode.LeftBracket)) Nudge(-0.1f);
        if (Input.GetKeyDown(KeyCode.RightBracket)) Nudge(+0.1f);

        // Space: 일시정지 토글
        if (Input.GetKeyDown(KeyCode.Space)) TogglePause();

        // 마우스 휠로 미세 조절
        float wheel = Input.GetAxis("Mouse ScrollWheel");
        if (Mathf.Abs(wheel) > 0.0001f)
            Nudge(wheel > 0 ? +0.1f : -0.1f);
    }

    // 슬라이더 GUI
    void OnGUI()
    {
        if (!showOnGUISlider) return;

        const float w = 320f;
        const float h = 70f;
        GUI.Box(new Rect(10, 10, w, h), "Time Scale");

        GUILayout.BeginArea(new Rect(20, 35, w - 20, h - 20));
        GUILayout.BeginHorizontal();
        GUILayout.Label($"{(IsPaused ? "Paused" : $"{CurrentTimeScale:0.00}x")}", GUILayout.Width(120));
        float newVal = GUILayout.HorizontalSlider(CurrentTimeScale, minTimeScale, maxTimeScale, GUILayout.Width(160));
        if (Mathf.Abs(newVal - CurrentTimeScale) > 0.0001f)
        {
            SetTimeScale(newVal);
            _targetScale = newVal;
        }
        if (GUILayout.Button(IsPaused ? "Resume" : "Pause", GUILayout.Width(60)))
            TogglePause();
        GUILayout.EndHorizontal();
        GUILayout.EndArea();
    }

    // ─── Public API ──────────────────────────────────────────────
    public void SetTimeScale(float scale)
    {
        StopLerp();
        _targetScale = Mathf.Clamp(scale, minTimeScale, maxTimeScale);
        ApplyTimeScale(_targetScale);
    }

    public void Pause() => ApplyTimeScale(0f);
    public void Resume() => ApplyTimeScale(Mathf.Clamp(_targetScale < minTimeScale ? 1f : _targetScale, minTimeScale, maxTimeScale));
    public void TogglePause()
    {
        if (IsPaused) Resume(); else Pause();
    }

    /// <summary>부드럽게 배속 변경 (unscaled 시간 기준)</summary>
    public void SmoothSetTimeScale(float targetScale, float duration)
    {
        StopLerp();
        targetScale = Mathf.Clamp(targetScale, minTimeScale, maxTimeScale);
        lerpRoutine = StartCoroutine(CoSmoothSet(targetScale, duration));
        _targetScale = targetScale;
    }

    // ─── Internal ───────────────────────────────────────────────
    private void Nudge(float delta)
    {
        if (IsPaused) return; // 일시정지 중에는 슬라이더 값만 바꾸지 않음
        SetTimeScale(Mathf.Clamp(CurrentTimeScale + delta, minTimeScale, maxTimeScale));
    }

    private void ApplyTimeScale(float scale)
    {
        float ts = Mathf.Max(0f, scale);
        Time.timeScale = ts;

        // ✅ 물리 적분 간격을 항상 고정(예: 0.02s)
        Time.fixedDeltaTime = baseFixedDeltaTime;

        onTimeScaleChanged?.Invoke(ts);
    }

    private IEnumerator CoSmoothSet(float targetScale, float duration)
    {
        float start = Time.timeScale;
        float t = 0f;
        duration = Mathf.Max(0f, duration);

        if (duration <= 0f)
        {
            ApplyTimeScale(targetScale);
            yield break;
        }

        while (t < 1f)
        {
            t += Time.unscaledDeltaTime / duration;
            float s = Mathf.Lerp(start, targetScale, t);
            ApplyTimeScale(s);
            yield return null;
        }
        lerpRoutine = null;
    }

    private void StopLerp()
    {
        if (lerpRoutine != null)
        {
            StopCoroutine(lerpRoutine);
            lerpRoutine = null;
        }
    }

    void OnDestroy()
    {
        // 종료 시 기본값 복구(에디터 편의). 필요 없으면 제거
        Time.timeScale = 1f;
        Time.fixedDeltaTime = baseFixedDeltaTime;
    }
}
