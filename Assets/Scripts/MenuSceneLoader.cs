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
    [SerializeField] private string dataFileScene = "dataFileScene";
    [SerializeField] private string selectModeScene = "SelectModeMenu";

    public void LoadEasy() => SceneManager.LoadScene(easyScene);
    public void LoadNormal() => SceneManager.LoadScene(normalScene);
    public void LoadHard() => SceneManager.LoadScene(hardScene);
    public void LoadTraining() => SceneManager.LoadScene(trainingScene);
    public void LoadAchievement() => SceneManager.LoadScene(achievementScene);
    public void LoadDataFile() => SceneManager.LoadScene(dataFileScene);
    public void LoadSelectMode() => SceneManager.LoadScene(selectModeScene);

}
