using System;
using System.Collections.Generic;
using UnityEngine;

public class EnemyShooting : MonoBehaviour
{
    [Header("Swing Zones")]
    public SwingZone overZone; // 모든 스윙은 overZone만 사용

    public List<Shuttlecock> shuttlecocksInRange = new List<Shuttlecock>();

    [Header("Test Launch Values")]
    public float testYaw = 0f;
    public float testPitch = 45f;
    public float testForce = 50f;

    void Update()
    {
        // 1~4 숫자 키 입력에 따른 4가지 스윙
        if (Input.GetKeyDown(KeyCode.Alpha1)) OverStrong();
        if (Input.GetKeyDown(KeyCode.Alpha2)) OverWeak();
        if (Input.GetKeyDown(KeyCode.Alpha3)) UnderStrong();
        if (Input.GetKeyDown(KeyCode.Alpha4)) UnderWeak();

        // Q를 누르면 Test 발사
        if (Input.GetKeyDown(KeyCode.Q))
        {
            Test();
        }
    }

    // Test: inspector에서 지정한 testYaw, testPitch, testForce로 발사
    void Test()
    {
        // 테스트는 현재 감지 목록 대상으로
        LaunchToAll(testYaw, testPitch, testForce, "Test", shuttlecocksInRange);
    }

    // 좌우 각도(yaw), 위아래 각도(pitch), 힘(force), 로그 이름, 타겟들
    void LaunchToAll(float baseYaw, float pitch, float force, string shotName, List<Shuttlecock> targets)
    {
        // PlayerShooting과 동일한 yaw 로직
        float playerX = transform.position.x;
        float yaw = 0f;

        if (playerX <= -5f)
        {
            float t = Mathf.InverseLerp(-10f, -5f, playerX);
            yaw = UnityEngine.Random.Range(0f, Mathf.Lerp(20f, 0f, t));
        }
        else if (playerX < 5f)
        {
            yaw = UnityEngine.Random.Range(-10f, 10f);
        }
        else
        {
            float t = Mathf.InverseLerp(5f, 10f, playerX);
            yaw = UnityEngine.Random.Range(Mathf.Lerp(-20f, 0f, t), 0f);
        }

        foreach (Shuttlecock sc in targets)
        {
            if (sc != null)
            {
                sc.Launch(yaw, pitch, force);
                Debug.Log($"{shotName} 발사됨 (Yaw: {yaw}, Pitch: {pitch}, Force: {force}) → {sc.name}");
            }
        }

        // 필요 시 비우기 (원래 EnemyShooting은 비웠음)
        shuttlecocksInRange.Clear();
    }

    // 아래 4개 스윙의 파라미터를 PlayerShooting과 1:1 일치
    public void OverStrong()
    {
        shuttlecocksInRange = overZone != null ? overZone.GetShuttlecocks() : new List<Shuttlecock>();
        if (shuttlecocksInRange.Count == 0) return; // 헛스윙: 아무 것도 안 함
        LaunchToAll(0f, 10f, 25f, "OverStrong", shuttlecocksInRange);
    }
    public void OverWeak()
    {
        shuttlecocksInRange = overZone != null ? overZone.GetShuttlecocks() : new List<Shuttlecock>();
        if (shuttlecocksInRange.Count == 0) return;
        LaunchToAll(0f, 30f, 20f, "OverWeak", shuttlecocksInRange);
    }
    public void UnderStrong()
    {
        shuttlecocksInRange = overZone != null ? overZone.GetShuttlecocks() : new List<Shuttlecock>();
        if (shuttlecocksInRange.Count == 0) return;
        LaunchToAll(0f, 45f, 20f, "UnderStrong", shuttlecocksInRange);
    }
    public void UnderWeak()
    {
        shuttlecocksInRange = overZone != null ? overZone.GetShuttlecocks() : new List<Shuttlecock>();
        if (shuttlecocksInRange.Count == 0) return;
        LaunchToAll(0f, 60f, 10f, "UnderWeak", shuttlecocksInRange);
    }
    
    /* (미사용) 트리거로 셔틀 목록 유지하던 로직 — 필요 시 재활성화 가능
    void OnTriggerEnter(Collider other)
    {
        Shuttlecock sc = other.GetComponent<Shuttlecock>();
        if (sc != null && !shuttlecocksInRange.Contains(sc))
        {
            shuttlecocksInRange.Add(sc);
        }
    }
    void OnTriggerExit(Collider other)
    {
        Shuttlecock sc = other.GetComponent<Shuttlecock>();
        if (sc != null && shuttlecocksInRange.Contains(sc))
        {
            shuttlecocksInRange.Remove(sc);
        }
    }*/
}
