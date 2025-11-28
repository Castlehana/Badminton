using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// 전역 오디오 매니저.
/// - 씬에 하나만 존재해야 하며, DontDestroyOnLoad 로 유지됩니다.
/// - 인스펙터에서 BGM / SFX 리스트를 설정해 두고, 다른 스크립트에서
///   AudioManager.Instance.PlayBGM("Key"), PlaySFX("Key") 로 사용하세요.
/// </summary>
public class AudioManager : MonoBehaviour
{
    public static AudioManager Instance { get; private set; }

    [System.Serializable]
    public class SoundData
    {
        public string key;
        public AudioClip clip;
        [Range(0f, 1f)] public float volume = 1f;
    }

    [Header("BGM")]
    public AudioSource bgmSource;
    public List<SoundData> bgmClips = new List<SoundData>();

    [Header("SFX")]
    public AudioSource sfxSource;
    public List<SoundData> sfxClips = new List<SoundData>();

    private Dictionary<string, SoundData> _bgmDict;
    private Dictionary<string, SoundData> _sfxDict;

    private void Awake()
    {
        if (Instance != null && Instance != this)
        {
            Destroy(gameObject);
            return;
        }

        Instance = this;
        DontDestroyOnLoad(gameObject);

        _bgmDict = new Dictionary<string, SoundData>();
        foreach (var s in bgmClips)
        {
            if (!string.IsNullOrEmpty(s.key) && s.clip != null)
            {
                _bgmDict[s.key] = s;
            }
        }

        _sfxDict = new Dictionary<string, SoundData>();
        foreach (var s in sfxClips)
        {
            if (!string.IsNullOrEmpty(s.key) && s.clip != null)
            {
                _sfxDict[s.key] = s;
            }
        }
    }

    // === BGM ===

    public void PlayBGM(string key, bool loop = true)
    {
        if (bgmSource == null)
        {
            Debug.LogWarning("[AudioManager] BGM Source 가 설정되어 있지 않습니다.");
            return;
        }

        if (_bgmDict == null || !_bgmDict.TryGetValue(key, out var data))
        {
            Debug.LogWarning($"[AudioManager] BGM key 를 찾을 수 없습니다: {key}");
            return;
        }

        bgmSource.clip = data.clip;
        bgmSource.volume = data.volume;
        bgmSource.loop = loop;
        bgmSource.Play();
    }

    public void StopBGM()
    {
        if (bgmSource != null)
        {
            bgmSource.Stop();
        }
    }

    // === SFX ===

    public void PlaySFX(string key, float volumeScale = 1f)
    {
        if (sfxSource == null)
        {
            Debug.LogWarning("[AudioManager] SFX Source 가 설정되어 있지 않습니다.");
            return;
        }

        if (_sfxDict == null || !_sfxDict.TryGetValue(key, out var data))
        {
            Debug.LogWarning($"[AudioManager] SFX key 를 찾을 수 없습니다: {key}");
            return;
        }

        sfxSource.PlayOneShot(data.clip, data.volume * volumeScale);
    }
}


