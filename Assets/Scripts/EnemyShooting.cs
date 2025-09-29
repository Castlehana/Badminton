using System;
using System.Collections.Generic;

using UnityEngine;

public class EnemyShooting : MonoBehaviour
{
    [Header("Swing Zones")]
    public SwingZone overZone;

    public List<Shuttlecock> shuttlecocksInRange = new List<Shuttlecock>();

    [Header("Test Launch Values")]
    public float testYaw = 0f;
    public float testPitch = 45f;
    public float testForce = 50f;


    void Update()
    {
        // 1~4 숫자 키 입력에 따른 4가지 스윙 재 구성
        if (Input.GetKeyDown(KeyCode.Alpha1)) OverStrong();
        if (Input.GetKeyDown(KeyCode.Alpha2)) OverWeak();
        if (Input.GetKeyDown(KeyCode.Alpha3)) UnderStrong();
        if (Input.GetKeyDown(KeyCode.Alpha4)) UnderWeak();

        //if (Input.GetKeyDown(KeyCode.Alpha1) || Input.GetKeyDown(KeyCode.Keypad1)) Clear();
        //if (Input.GetKeyDown(KeyCode.Alpha2) || Input.GetKeyDown(KeyCode.Keypad2)) Drop();
        //if (Input.GetKeyDown(KeyCode.Alpha3) || Input.GetKeyDown(KeyCode.Keypad3)) Smash();
        //if (Input.GetKeyDown(KeyCode.Alpha4) || Input.GetKeyDown(KeyCode.Keypad4)) Push();
        //if (Input.GetKeyDown(KeyCode.Alpha5) || Input.GetKeyDown(KeyCode.Keypad5)) Hairpin();
        //if (Input.GetKeyDown(KeyCode.Alpha6) || Input.GetKeyDown(KeyCode.Keypad6)) Drive();
        //if (Input.GetKeyDown(KeyCode.Alpha7) || Input.GetKeyDown(KeyCode.Keypad7)) Serve();

        // Q를 누르면 Test 발사
        if (Input.GetKeyDown(KeyCode.Q))
        {
            Test();
        }
    }

    // Test: inspector에서 지정한 testYaw, testPitch, testForce로 발사
    void Test()
    {
        LaunchToAll(testYaw, testPitch, testForce, "Test");
    }

    //좌우 각도, 위아래 각도, 힘, 이
    void LaunchToAll(float baseYaw, float pitch, float force, string shotName)
    {
        pitch = 180-pitch;
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

        foreach (Shuttlecock sc in shuttlecocksInRange)
        {
            if (sc != null)
            {
                sc.Launch(yaw, pitch, force);
                Debug.Log($"{shotName} 발사됨 (Yaw: {yaw}) → {sc.name}");
            }
        }

        shuttlecocksInRange.Clear();
    }

    public void Serve() => LaunchToAll(0f, 45f, 15f, "서비스");

    public void OverStrong()
    {
        shuttlecocksInRange = overZone.GetShuttlecocks();
        if (shuttlecocksInRange.Count == 0)
        {
            // 없는데 휘두름(헛스윙)
            return;
        }

        LaunchToAll(0f, -5f, 40f, "OverStrong");
    }
    public void OverWeak()
    {
        shuttlecocksInRange = overZone.GetShuttlecocks();
        if (shuttlecocksInRange.Count == 0)
        {
            // 없는데 휘두름(헛스윙)
            return;
        }

        LaunchToAll(0f, 45f, 25f, "OverWeak");
    }
    public void UnderStrong()
    {
        shuttlecocksInRange = overZone.GetShuttlecocks();
        if (shuttlecocksInRange.Count == 0)
        {
            // 없는데 휘두름(헛스윙)
            return;
        }

        LaunchToAll(0f, 60f, 10f, "UnderStrong");
    }
    public void UnderWeak()
    {
        shuttlecocksInRange = overZone.GetShuttlecocks();
        if (shuttlecocksInRange.Count == 0)
        {
            // 없는데 휘두름(헛스윙)
            return;
        }

        LaunchToAll(0f, 45f, 15f, "UnderWeak");
    }

    // 사용하지 않게됨
    void OnTriggerEnter(Collider other)
    {
        Shuttlecock sc = other.GetComponent<Shuttlecock>();
        if (sc != null && !shuttlecocksInRange.Contains(sc))
        {
            shuttlecocksInRange.Add(sc);
            //Debug.Log($"셔틀콕 감지됨: {sc.name} 트리거 안에 들어옴");
        }
    }
    void OnTriggerExit(Collider other)
    {
        Shuttlecock sc = other.GetComponent<Shuttlecock>();
        if (sc != null && shuttlecocksInRange.Contains(sc))
        {
            shuttlecocksInRange.Remove(sc);
            //Debug.Log($"셔틀콕 나감: {sc.name} 트리거 밖으로 나감");
        }
    }
}
