using System;
using System.Collections.Generic;
using UnityEngine;

public class EnemyShooting : MonoBehaviour
{
    [Header("Swing Zones")]
    public SwingZone overZone; // 오버 스윙 (Clear, Drop)용
    public SwingZone underZone; // 언더 스윙 (Hairpin, Drive, Under)용

    public List<Shuttlecock> shuttlecocksInRange = new List<Shuttlecock>();

    [Header("Test Launch Values")]
    public float testYaw = 0f;
    public float testPitch = 45f;
    public float testForce = 50f;

    [Header("Shot Variation")]
    public float yawJitterDeg = 0f; // 0이면 고정 각도, >0이면 ±범위로 약간의 분산

    void Update()
    {
        // 1~4 숫자 키 입력에 따른 4가지 스윙
        //if (Input.GetKeyDown(KeyCode.Alpha1)) OverStrong();
        //if (Input.GetKeyDown(KeyCode.Alpha2)) OverWeak();
        //if (Input.GetKeyDown(KeyCode.Alpha3)) UnderStrong();
        //if (Input.GetKeyDown(KeyCode.Alpha4)) UnderWeak();

        //********** 추가 *********** 위에 스윙 함수들을 아래 함수들로 사용하시오
        if (Input.GetKeyDown(KeyCode.Alpha1) || Input.GetKeyDown(KeyCode.Keypad1)) Clear();
        if (Input.GetKeyDown(KeyCode.Alpha2) || Input.GetKeyDown(KeyCode.Keypad2)) Drop();
        if (Input.GetKeyDown(KeyCode.Alpha3) || Input.GetKeyDown(KeyCode.Keypad3)) Hairpin();
        if (Input.GetKeyDown(KeyCode.Alpha4) || Input.GetKeyDown(KeyCode.Keypad4)) Drive();
        if (Input.GetKeyDown(KeyCode.Alpha5) || Input.GetKeyDown(KeyCode.Keypad5)) Under();
        if (Input.GetKeyDown(KeyCode.Alpha6) || Input.GetKeyDown(KeyCode.Keypad6)) Smash();



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
            yaw = UnityEngine.Random.Range(0f, Mathf.Lerp(5f, 0f, t));
        }
        else if (playerX < 5f)
        {
            yaw = UnityEngine.Random.Range(-10f, 10f);
        }
        else
        {
            float t = Mathf.InverseLerp(5f, 10f, playerX);
            yaw = UnityEngine.Random.Range(Mathf.Lerp(-5f, 0f, t), 0f);
        }

        // 첫 번째 셔틀에 대해서만 로그 출력
        bool firstLogged = false;
        foreach (Shuttlecock sc in targets)
        {
            if (sc != null)
            {
                sc.Launch(yaw, pitch, force); 
                {
                    Debug.Log($"{shotName} 발사됨 (Yaw: {yaw}, Pitch: {pitch}, Force: {force}) → {sc.name}");
                    firstLogged = true;
                }
            }
        }
    }

    // 오버 스윙: Clear, Drop, Drive
    public void Clear()
    {
        var targets = overZone != null ? new List<Shuttlecock>(overZone.GetShuttlecocks()) : new List<Shuttlecock>();
        if (targets.Count == 0) return;
        LaunchToAll(0f, 45f, 35f, "클리어", targets);
    }
    
    public void Drop()
    {
        var targets = overZone != null ? new List<Shuttlecock>(overZone.GetShuttlecocks()) : new List<Shuttlecock>();
        if (targets.Count == 0) return;
        LaunchToAll(0f, 50f, 15f, "드롭", targets);
    }
    
        public void Drive()
    {
        var targets = overZone != null ? new List<Shuttlecock>(overZone.GetShuttlecocks()) : new List<Shuttlecock>();
        if (targets.Count == 0) return;
        LaunchToAll(0f, 10f, 25f, "드라이브", targets);
    }
    

    // 언더 스윙: Hairpin, Under
    public void Hairpin()
    {
        var targets = underZone != null ? new List<Shuttlecock>(underZone.GetShuttlecocks()) : new List<Shuttlecock>();
        if (targets.Count == 0) return;
        LaunchToAll(0f, 35f, 13f, "헤어핀", targets);
    }
    
    public void Under()
    {
        var targets = underZone != null ? new List<Shuttlecock>(underZone.GetShuttlecocks()) : new List<Shuttlecock>();
        if (targets.Count == 0) return;
        LaunchToAll(0f, 30f, 20f, "언더", targets);
    }

    public void Smash()
    {
        var list = overZone.GetShuttlecocks();
        var targets = new List<Shuttlecock>(list);
        if (targets.Count == 0)
        {
            Debug.Log("[EnemyShooting] Smash FAILED: no targets in overZone");
            return;
        }

        LaunchToAll(0f, -5f, 30f, "스매시!!!!!!!!!!!!", targets);
    }


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
