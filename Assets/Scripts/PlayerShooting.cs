using System;
using System.Collections.Generic;
using System.Security.Cryptography;
using UnityEngine;

public class PlayerShooting : MonoBehaviour
{
    [Header("Swing Zones")]
    public SwingZone overZone;
    public SwingZone underZone;

    public List<Shuttlecock> shuttlecocksInRange = new List<Shuttlecock>();

    [Header("Test Launch Values")]
    public float testYaw = 0f;
    public float testPitch = 45f;
    public float testForce = 50f;

    public RallyManager rallyManager;
    public GameObject shuttlePrefab;
    public Transform spawnPoint;
    public AutoMovement player;

    void Update()
    {
        ////1~4 숫자 키 입력에 따른 4가지 스윙 재 구성
        //if (Input.GetKeyDown(KeyCode.Alpha1) || Input.GetKeyDown(KeyCode.Keypad1)) OverStrong();
        //if (Input.GetKeyDown(KeyCode.Alpha2) || Input.GetKeyDown(KeyCode.Keypad2)) OverWeak();
        //if (Input.GetKeyDown(KeyCode.Alpha3) || Input.GetKeyDown(KeyCode.Keypad3)) UnderStrong();
        //if (Input.GetKeyDown(KeyCode.Alpha4) || Input.GetKeyDown(KeyCode.Keypad4)) UnderWeak();

        if (Input.GetKeyDown(KeyCode.Alpha1) || Input.GetKeyDown(KeyCode.Keypad1)) ClearSwing();
        if (Input.GetKeyDown(KeyCode.Alpha2) || Input.GetKeyDown(KeyCode.Keypad2)) DropSwing();
        if (Input.GetKeyDown(KeyCode.Alpha3) || Input.GetKeyDown(KeyCode.Keypad3)) HairpinSwing();
        if (Input.GetKeyDown(KeyCode.Alpha4) || Input.GetKeyDown(KeyCode.Keypad4)) DriveSwing();
        if (Input.GetKeyDown(KeyCode.Alpha5) || Input.GetKeyDown(KeyCode.Keypad5)) UnderSwing();

        // Q를 누르면 Test 발사
        if (Input.GetKeyDown(KeyCode.Q))
        {
            Test();
        }
    }

    // Test: inspector에서 지정한 testYaw, testPitch, testForce로 발사
    void Test()
    {
        LaunchToAll(testYaw, testPitch, testForce, "Test", shuttlecocksInRange);
    }

    void LaunchToAll(float baseYaw, float pitch, float force, string shotName, List<Shuttlecock> targets, bool markAsSmash = false)
    {
        float playerX = transform.position.x;
        float yaw = 0f;

        if (playerX <= -5f)
        {
            float t = Mathf.InverseLerp(5f, 10f, playerX);
            yaw = UnityEngine.Random.Range(Mathf.Lerp(-20f, 0f, t), 0f);
        }
        else if (playerX < 5f)
        {
            yaw = UnityEngine.Random.Range(-10f, 10f);
        }
        else
        {
            float t = Mathf.InverseLerp(-10f, -5f, playerX);
            yaw = UnityEngine.Random.Range(0f, Mathf.Lerp(20f, 0f, t));
        }
        UnityEngine.Debug.Log("런치투올 ㅋ");
        foreach (Shuttlecock sc in targets)
        {
            UnityEngine.Debug.Log("포이치드렁옴");
            if (sc != null)
            {
                UnityEngine.Debug.Log("이프 들어옴");
                if (sc.Launch(yaw, pitch, force))
                {
                    PlaySwingSfx(markAsSmash);
                    Debug.Log($"{shotName} 발사됨 (Yaw: {yaw}) → {sc.name}");
                }
            }
        }
    }

    private void PlaySwingSfx(bool isSmash)
    {
        if (AudioManager.Instance == null) return;

        string key = isSmash ? "Shuttle_Smash" : "Shuttle_Hit";
        AudioManager.Instance.PlaySFX(key);
    }

    public void Clear() => LaunchToAll(0f, 180-45f, 30f, "클리어", shuttlecocksInRange);
    public void Drop() => LaunchToAll(0f, 180 - 50f, 15f, "드롭", shuttlecocksInRange);
    public void Smash() => LaunchToAll(0f, 180 + 5f, 30f, "스매시", shuttlecocksInRange, true);
 
    //public void Push() => LaunchToAll(0f, -40f, 40f, "푸시", shuttlecocksInRange);
    public void Hairpin() => LaunchToAll(0f, 180 - 38f, 18f, "헤어핀", shuttlecocksInRange);
    public void Drive() => LaunchToAll(0f, 180 - 10f, 25f, "드라이브", shuttlecocksInRange);
    public void Serve() => LaunchToAll(0f, 180 - 45f, 20f, "서비스", shuttlecocksInRange);
    public void Under() => LaunchToAll(0f, 180 - 30f, 23f, "언더", shuttlecocksInRange);

    public void DriveSwing()
    {
        var targets = new List<Shuttlecock>(overZone.GetShuttlecocks()); // 복사
        if (targets.Count == 0) return;

        // 점프 중 발동 시 스매시로 구분
        if (player.isJumping)
        {
            Smash();
        }
        else
        {
            Drive();
        }
        var mgr = SaveManager.Instance;
        if (mgr != null && mgr.Current != null)
        {
            mgr.Current.achv.overSwing++;
            mgr.Save();
        }
    }

    public void ClearSwing() // OverStrong -> ClearSwing
    {
        var targets = new List<Shuttlecock>(overZone.GetShuttlecocks()); // 복사
        if (targets.Count == 0) return;

        // 점프 중 발동 시 스매시로 구분
        if (player.isJumping)
        {
            Smash();
        }
        else
        {
            Clear();
        }
        var mgr = SaveManager.Instance;
        if (mgr != null && mgr.Current != null)
        {
            mgr.Current.achv.overSwing++;
            mgr.Save();
        }
    }

    
    public void DropSwing() // OverWeak -> DropSwing
    {
        var targets = new List<Shuttlecock>(overZone.GetShuttlecocks()); // 복사
        if (targets.Count == 0) return;

        Drop();
        var mgr = SaveManager.Instance;
        if (mgr != null && mgr.Current != null)
        {
            mgr.Current.achv.overSwing++;
            mgr.Save();
        }
    }

    //public void OverWeak()
    //{
    //    var targets = new List<Shuttlecock>(overZone.GetShuttlecocks()); // 복사
    //    if (targets.Count == 0) return;

    //    LaunchToAll(0f, 180f-45f, 15f, "OverWeak", targets);
    //}

    public void UnderSwing() //UnderString -> UnderSwing
    {
        // 점프중엔 언더로 못침
        if (player.isJumping) return;

        // 서브의 경우
        if ((rallyManager.State == RallyState.Ready) && (rallyManager.Turn == ServeTurn.MyTurn))
        {
            UnityEngine.Debug.Log("서브!!");
            GameObject newShuttle = Instantiate(shuttlePrefab, spawnPoint.position, Quaternion.identity);
            Shuttlecock shuttle = newShuttle.GetComponent<Shuttlecock>();

            shuttle.Launch(0f, 180 - 45f, 20f);

            rallyManager.State = RallyState.Rallying;

            return;
        }

        var targets = new List<Shuttlecock>(underZone.GetShuttlecocks()); // 복사
        if (targets.Count == 0)
        {
            return;
        }
        Under();
        var mgr = SaveManager.Instance;
        if (mgr != null && mgr.Current != null)
        {
            mgr.Current.achv.underSwing++;
            mgr.Save();
        }
    }

    public void HairpinSwing() //UnderWeak -> HairpinSwing
    {
        // 점프중엔 언더로 못침
        if (player.isJumping) return;

        // 서브의 경우
        if ((rallyManager.State == RallyState.Ready) && (rallyManager.Turn == ServeTurn.MyTurn))
        {
            UnityEngine.Debug.Log("서브!!");
            GameObject newShuttle = Instantiate(shuttlePrefab, spawnPoint.position, Quaternion.identity);
            Shuttlecock shuttle = newShuttle.GetComponent<Shuttlecock>();

            shuttle.Launch(0f, 180f - 45f, 20f);

            rallyManager.State = RallyState.Rallying;

            return;
        }

        var targets = new List<Shuttlecock>(underZone.GetShuttlecocks()); // 복사
        if (targets.Count == 0) return;

        Hairpin();
        var mgr = SaveManager.Instance;
        if (mgr != null && mgr.Current != null)
        {
            mgr.Current.achv.underSwing++;
            mgr.Save();
        }
    }

    //public void UnderWeak()
    //{
    //    // 점프중엔 언더로 못침
    //    if (player.isJumping) return;

    //    // 서브의 경우
    //    if (rallyManager.State == RallyState.Ready)
    //    {
    //        UnityEngine.Debug.Log("서브!!");
    //        GameObject newShuttle = Instantiate(shuttlePrefab, spawnPoint.position, Quaternion.identity);
    //        Shuttlecock shuttle = newShuttle.GetComponent<Shuttlecock>();

    //        shuttle.Launch(0f, 180f - 45f, 20f);

    //        rallyManager.State = RallyState.Rallying;

    //        return;
    //    }

    //    var targets = new List<Shuttlecock>(underZone.GetShuttlecocks()); // 복사
    //    if (targets.Count == 0) return;

    //    LaunchToAll(0f, 180f - 60f, 15f, "UnderWeak", targets);
    //}

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
        if (!other.CompareTag("SwingZone")) return;

        Shuttlecock sc = other.GetComponent<Shuttlecock>();
        if (sc != null && shuttlecocksInRange.Contains(sc))
            shuttlecocksInRange.Remove(sc);
    }
}
