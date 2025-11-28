using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

public class MenuSceneLoader : MonoBehaviour
{
    [Header("Scene name")]
    [SerializeField] private string easyScene = "Easy_GameMap";
    [SerializeField] private string normalScene = "normal_GameMap";
    [SerializeField] private string hardScene = "Hard_GameMap";
    [SerializeField] private string trainingScene = "Training_GameMap";
    [SerializeField] private string achievementScene = "AchievementScene";
    [SerializeField] private string dataFileScene = "SaveDataMenu";
    [SerializeField] private string selectModeScene = "SelectModeMenu";
    [SerializeField] private string winScene = "WinScene";
    [SerializeField] private string loseScene = "LoseScene";
    [SerializeField] private string titleScene = "TitleScene";

    [Header("Fade Controller")]
    public SceneFader sceneFader;   // 인스펙터에서 할당

    // 공통: 페이드 있으면 페이드로, 없으면 바로 로드
    private void LoadSceneWithFade(string sceneName)
    {
        if (sceneFader != null)
        {
            sceneFader.FadeOutAndLoadScene(sceneName);
        }
        else
        {
            SceneManager.LoadScene(sceneName);
        }
    }

    public void LoadEasy() => LoadSceneWithFade(easyScene);
    public void LoadNormal() => LoadSceneWithFade(normalScene);
    public void LoadHard() => LoadSceneWithFade(hardScene);
    public void LoadTraining() => LoadSceneWithFade(trainingScene);
    public void LoadAchievement() => LoadSceneWithFade(achievementScene);
    public void LoadDataFile() => LoadSceneWithFade(dataFileScene);
    public void LoadSelectMode() => LoadSceneWithFade(selectModeScene);
    public void LoadWinScene() => LoadSceneWithFade(winScene);
    public void LoadLoseScene() => LoadSceneWithFade(loseScene);
    public void LoadTitleScene() => LoadSceneWithFade(titleScene);
}
