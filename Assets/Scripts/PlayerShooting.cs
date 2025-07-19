using UnityEngine;
using System.Collections.Generic;

public class PlayerShooting : MonoBehaviour
{
    private List<Shuttlecock> shuttlecocksInRange = new List<Shuttlecock>();

    void Update()
    {
        if (Input.GetKeyDown(KeyCode.Alpha1) || Input.GetKeyDown(KeyCode.Keypad1)) Clear();
        if (Input.GetKeyDown(KeyCode.Alpha2) || Input.GetKeyDown(KeyCode.Keypad2)) Drop();
        if (Input.GetKeyDown(KeyCode.Alpha3) || Input.GetKeyDown(KeyCode.Keypad3)) Smash();
        if (Input.GetKeyDown(KeyCode.Alpha4) || Input.GetKeyDown(KeyCode.Keypad4)) Push();
        if (Input.GetKeyDown(KeyCode.Alpha5) || Input.GetKeyDown(KeyCode.Keypad5)) Hairpin();
        if (Input.GetKeyDown(KeyCode.Alpha6) || Input.GetKeyDown(KeyCode.Keypad6)) Drive();
    }

    void LaunchToAll(float yaw, float pitch, float force, string shotName)
    {
        foreach (Shuttlecock sc in shuttlecocksInRange)
        {
            if (sc != null)
            {
                sc.Launch(yaw, pitch, force);
                UnityEngine.Debug.Log($"{shotName} 발사됨 → {sc.name}");
            }
        }

        shuttlecocksInRange.Clear(); // 한 번 발사한 셔틀콕은 리스트에서 제거
    }

    void Clear() => LaunchToAll(0f, 45f, 60f, "클리어");
    void Drop() => LaunchToAll(0f, 60f, 30f, "드롭");
    void Smash() => LaunchToAll(0f, -5f, 50f, "스매시");
    void Push() => LaunchToAll(0f, -15f, 50f, "푸시");
    void Hairpin() => LaunchToAll(0f, 45f, 5f, "헤어핀");
    void Drive() => LaunchToAll(0f, 5f, 70f, "드라이브");

    void OnTriggerEnter(Collider other)
    {
        Shuttlecock sc = other.GetComponent<Shuttlecock>();
        if (sc != null && !shuttlecocksInRange.Contains(sc))
        {
            shuttlecocksInRange.Add(sc);
            UnityEngine.Debug.Log($"셔틀콕 감지됨: {sc.name} 트리거 안에 들어옴");
        }
    }

    void OnTriggerExit(Collider other)
    {
        Shuttlecock sc = other.GetComponent<Shuttlecock>();
        if (sc != null && shuttlecocksInRange.Contains(sc))
        {
            shuttlecocksInRange.Remove(sc);
            UnityEngine.Debug.Log($"셔틀콕 나감: {sc.name} 트리거 밖으로 나감");
        }
    }
}
