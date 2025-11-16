using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.SceneManagement;

public class SaveManager : MonoBehaviour
{
    public static SaveManager Instance { get; private set; }
    public SaveData Current;
    public int CurrentSlot => Current?.slotIndex ?? 0;

    // 파일 경로: <persistent>/save_slot_{i}.json
    public static string GetPath(int slot)
    {
        return Path.Combine(Application.persistentDataPath, $"save_slot{slot}.json");
    }

    void Awake()
    {
        if (Instance != null && Instance != this) { Destroy(gameObject); return; }
        Instance = this;
        DontDestroyOnLoad(gameObject);
    }

    // 슬롯 존재 여부
    public static bool Exists(int slot) => File.Exists(GetPath(slot));

    // 슬롯 요약(버튼 라벨에 쓰기 좋게)
    public static string GetSummaryText(int slot)
    {
        if (!Exists(slot)) return "Empty Slot";
        try
        {
            var json = File.ReadAllText(GetPath(slot));
            var data = JsonUtility.FromJson<SaveData>(json);
            return $"{data.profileName} | Wins {data.achv.totalWins} / Loses {data.achv.totalLoses}";
        }
        catch { return "Corruted Slot"; }
    }

    // 새로운 데이터 생성
    public void CreateNew(int slot, string name = "Player")
    {
        Current = new SaveData(slot, name);
        Save(); // 즉시 저장
    }

    // 세이브 데이터 로드
    public bool Load(int slot)
    {
        var path = GetPath(slot);
        if (!File.Exists(path)) return false;
        try
        {
            var json = File.ReadAllText(path);
            Current = JsonUtility.FromJson<SaveData>(json);
            return true;
        }
        catch
        {
            Current = null;
            return false;
        }
    }

    // 세이브 데이터 저장
    public void Save()
    {
        if (Current == null) return;
        Current.TouchUpdatedAt();
        var json = JsonUtility.ToJson(Current, true);
        File.WriteAllText(GetPath(Current.slotIndex), json);
    }

    // 삭제 (슬롯 비우기)
    public static void Delete(int slot)
    {
        var path = GetPath(slot);
        if (File.Exists(path)) File.Delete(path);
    }

    // 안전 저장: 일시정지/종료 시
    void OnApplicationPause(bool pause) { if (pause) Save(); }
    void OnApplicationQuit() { Save(); }
}
