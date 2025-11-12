using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

// 세이브 파일의 데이터 모델

[Serializable]
public class AchievementData
{
    public int totalWins;
    public int totalLoses;
    // 계속 추가
}

public class SaveData
{
    public int slotIndex;           // 데이터 슬롯 인덱스
    public string profileName;      // 슬롯의 이름 (옵션)
    public AchievementData achv = new AchievementData();    // 통계 데이터 묶음

    public string lastScene = "ModeSelect"; // 마지막 씬 (옵션)
    public string createdAt;    // ISO-8601 (?)
    public string updatedAt;

    public SaveData(int slot, string name = "Player")
    {
        slotIndex = slot;
        profileName = name;

        createdAt = DateTime.UtcNow.ToString("o");
        updatedAt = createdAt;
    }

    public void TouchUpdatedAt() => updatedAt = DateTime.UtcNow.ToString("o");
}
