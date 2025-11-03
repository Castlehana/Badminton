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

        //********** 추가 *********** 위에 스윙 함수들을 아래 함수들로 사용하시오
        //if (Input.GetKeyDown(KeyCode.Alpha1) || Input.GetKeyDown(KeyCode.Keypad1)) Clear();
        //if (Input.GetKeyDown(KeyCode.Alpha2) || Input.GetKeyDown(KeyCode.Keypad2)) Drop();
        //if (Input.GetKeyDown(KeyCode.Alpha3) || Input.GetKeyDown(KeyCode.Keypad3)) Hairpin();
        //if (Input.GetKeyDown(KeyCode.Alpha4) || Input.GetKeyDown(KeyCode.Keypad4)) Drive();
        //if (Input.GetKeyDown(KeyCode.Alpha5) || Input.GetKeyDown(KeyCode.Keypad5)) Under();
        ////if (Input.GetKeyDown(KeyCode.Alpha6) || Input.GetKeyDown(KeyCode.Keypad6)) Smash();



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
                sc.Launch(yaw, pitch, force); // ★ pitch 뒤집기 제거 → PlayerShooting과 동일
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
        LaunchToAll(0f, 40f, 15f, "OverStrong", shuttlecocksInRange);
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

    //********** 추가 *********** 위에 스윙 함수들을 아래 함수들로 사용하시오
    //public void Clear() => LaunchToAll(0f, 45f, 35f, "클리어", shuttlecocksInRange);
    //public void Drop() => LaunchToAll(0f, 50f, 15f, "드롭", shuttlecocksInRange);
    //public void Smash() => LaunchToAll(0f, -5f, 30f, "스매시", shuttlecocksInRange);
    //public void Hairpin() => LaunchToAll(0f, 35f, 13f, "헤어핀", shuttlecocksInRange);
    //public void Drive() => LaunchToAll(0f, 10f, 25f, "드라이브", shuttlecocksInRange);
    //public void Serve() => LaunchToAll(0f, 45f, 20f, "서비스", shuttlecocksInRange);
    //public void Under() => LaunchToAll(0f, 30f, 20f, "언더", shuttlecocksInRange);



    // (미사용) 트리거로 셔틀 목록 유지하던 로직 — 필요 시 재활성화 가능
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
    }
}
