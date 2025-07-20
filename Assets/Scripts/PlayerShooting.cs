using System;
using System.Collections.Generic;

using UnityEngine;

public class PlayerShooting : MonoBehaviour
{
    private List<Shuttlecock> shuttlecocksInRange = new List<Shuttlecock>();

    [Header("Test Launch Values")]
    public float testYaw = 0f;
    public float testPitch = 45f;
    public float testForce = 50f;

    void Update()
    {
        if (Input.GetKeyDown(KeyCode.Alpha1) || Input.GetKeyDown(KeyCode.Keypad1)) Clear();
        if (Input.GetKeyDown(KeyCode.Alpha2) || Input.GetKeyDown(KeyCode.Keypad2)) Drop();
        if (Input.GetKeyDown(KeyCode.Alpha3) || Input.GetKeyDown(KeyCode.Keypad3)) Smash();
        if (Input.GetKeyDown(KeyCode.Alpha4) || Input.GetKeyDown(KeyCode.Keypad4)) Push();
        if (Input.GetKeyDown(KeyCode.Alpha5) || Input.GetKeyDown(KeyCode.Keypad5)) Hairpin();
        if (Input.GetKeyDown(KeyCode.Alpha6) || Input.GetKeyDown(KeyCode.Keypad6)) Drive();

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

    void LaunchToAll(float baseYaw, float pitch, float force, string shotName)
    {
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

    void Clear() => LaunchToAll(0f, 45f, 35f, "클리어");
    void Drop() => LaunchToAll(0f, 60f, 15f, "드롭");
    void Smash() => LaunchToAll(0f, -5f, 40f, "스매시");
    void Push() => LaunchToAll(0f, -40f, 40f, "푸시");
    void Hairpin() => LaunchToAll(0f, 60f, 9f, "헤어핀");
    void Drive() => LaunchToAll(0f, 10f, 25f, "드라이브");

    void OnTriggerEnter(Collider other)
    {
        Shuttlecock sc = other.GetComponent<Shuttlecock>();
        if (sc != null && !shuttlecocksInRange.Contains(sc))
        {
            shuttlecocksInRange.Add(sc);
            Debug.Log($"셔틀콕 감지됨: {sc.name} 트리거 안에 들어옴");
        }
    }

    void OnTriggerExit(Collider other)
    {
        Shuttlecock sc = other.GetComponent<Shuttlecock>();
        if (sc != null && shuttlecocksInRange.Contains(sc))
        {
            shuttlecocksInRange.Remove(sc);
            Debug.Log($"셔틀콕 나감: {sc.name} 트리거 밖으로 나감");
        }
    }
}
