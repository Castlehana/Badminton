using UnityEngine;

public class SceneBGMPlayer : MonoBehaviour
{
    public string bgmKey = "알맞은 키 추가하셈";
    public bool loop = true;

    private void Start()
    {
        if (AudioManager.Instance != null && !string.IsNullOrEmpty(bgmKey))
        {
            AudioManager.Instance.PlayBGM(bgmKey, loop);
        }
    }
}


