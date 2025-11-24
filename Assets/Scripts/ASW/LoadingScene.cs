using System.Collections;
using UnityEngine;

public class LoadingScene : MonoBehaviour
{
    // 인스펙터에서 난이도 선택용 enum
    public enum Difficulty
    {
        Easy,
        Normal,
        Hard
    }

    [Header("Loading UI")]
    public GameObject LoadingUI;
    public float rotationSpeed = 100f;

    [Header("Scene Fader")]
    public GameObject sceneFaderObject;

    [Header("Target Game Scene")]
    public Difficulty targetDifficulty = Difficulty.Easy;   // 인스펙터에서 Easy / Normal / Hard 선택

    void Start()
    {
        StartCoroutine(CallSceneFaderFunctionAfterDelay(5f));
    }

    void Update()
    {
        if (LoadingUI != null)
        {
            LoadingUI.transform.Rotate(-1 * Vector3.forward * rotationSpeed * Time.deltaTime);
        }
    }

    IEnumerator CallSceneFaderFunctionAfterDelay(float delay)
    {
        yield return new WaitForSeconds(delay);

        if (sceneFaderObject != null)
        {
            SceneFader sceneFader = sceneFaderObject.GetComponent<SceneFader>();
            if (sceneFader != null)
            {
                // 난이도에 따라 씬 이름 매핑
                string nextSceneName = GetSceneNameByDifficulty(targetDifficulty);
                sceneFader.FadeOutAndLoadScene(nextSceneName);
            }
            else
            {
                Debug.LogWarning("SceneFader 스크립트를 찾을 수 없습니다.");
            }
        }
        else
        {
            Debug.LogWarning("SceneFader 오브젝트가 할당되지 않았습니다.");
        }
    }

    // enum → 실제 씬 이름 매핑
    string GetSceneNameByDifficulty(Difficulty difficulty)
    {
        switch (difficulty)
        {
            case Difficulty.Easy:
                return "Easy_GameMap 1";
            case Difficulty.Normal:
                return "Normal_GameMap";
            case Difficulty.Hard:
                return "Hard_GameMap 3";
            default:
                return "SelectModeMenu";
        }
    }
}
