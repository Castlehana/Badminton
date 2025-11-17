using System;
using System.Collections.Generic;
using UnityEngine;

public class EnemyShooting : MonoBehaviour
{
    [Header("Swing Zones")]
    public SwingZone overZone; // �ㅻ쾭 �ㅼ쐷 (Clear, Drop)��
    public SwingZone underZone; // �몃뜑 �ㅼ쐷 (Hairpin, Drive, Under)��

    public List<Shuttlecock> shuttlecocksInRange = new List<Shuttlecock>();

    [Header("Test Launch Values")]
    public float testYaw = 0f;
    public float testPitch = 45f;
    public float testForce = 50f;

    [Header("Shot Variation")]
    public float yawJitterDeg = 0f; // 0�대㈃ 怨좎젙 媛곷룄, >0�대㈃ 짹踰붿쐞濡� �쎄컙�� 遺꾩궛

    void Update()
    {
        // 1~4 �レ옄 �� �낅젰�� �곕Ⅸ 4媛�吏� �ㅼ쐷
        //if (Input.GetKeyDown(KeyCode.Alpha1)) OverStrong();
        //if (Input.GetKeyDown(KeyCode.Alpha2)) OverWeak();
        //if (Input.GetKeyDown(KeyCode.Alpha3)) UnderStrong();
        //if (Input.GetKeyDown(KeyCode.Alpha4)) UnderWeak();

        //********** 異붽� *********** �꾩뿉 �ㅼ쐷 �⑥닔�ㅼ쓣 �꾨옒 �⑥닔�ㅻ줈 �ъ슜�섏떆��
        if (Input.GetKeyDown(KeyCode.Alpha1) || Input.GetKeyDown(KeyCode.Keypad1)) Clear();
        if (Input.GetKeyDown(KeyCode.Alpha2) || Input.GetKeyDown(KeyCode.Keypad2)) Drop();
        if (Input.GetKeyDown(KeyCode.Alpha3) || Input.GetKeyDown(KeyCode.Keypad3)) Hairpin();
        if (Input.GetKeyDown(KeyCode.Alpha4) || Input.GetKeyDown(KeyCode.Keypad4)) Drive();
        if (Input.GetKeyDown(KeyCode.Alpha5) || Input.GetKeyDown(KeyCode.Keypad5)) Under();
        if (Input.GetKeyDown(KeyCode.Alpha6) || Input.GetKeyDown(KeyCode.Keypad6)) Smash();



        // Q瑜� �꾨Ⅴ硫� Test 諛쒖궗
        if (Input.GetKeyDown(KeyCode.Q))
        {
            Test();
        }
    }

    // Test: inspector�먯꽌 吏��뺥븳 testYaw, testPitch, testForce濡� 諛쒖궗
    void Test()
    {
        // �뚯뒪�몃뒗 �꾩옱 媛먯� 紐⑸줉 ���곸쑝濡�
        LaunchToAll(testYaw, testPitch, testForce, "Test", shuttlecocksInRange);
    }

    // 醫뚯슦 媛곷룄(yaw), �꾩븘�� 媛곷룄(pitch), ��(force), 濡쒓렇 �대쫫, ��寃잙뱾
    void LaunchToAll(float baseYaw, float pitch, float force, string shotName, List<Shuttlecock> targets)
    {
        // PlayerShooting怨� �숈씪�� yaw 濡쒖쭅
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

        // 泥� 踰덉㎏ �뷀��� ���댁꽌留� 濡쒓렇 異쒕젰
        bool firstLogged = false;
        foreach (Shuttlecock sc in targets)
        {
            if (sc != null)
            {
                sc.Launch(yaw, pitch, force); // �� pitch �ㅼ쭛湲� �쒓굅 �� PlayerShooting怨� �숈씪
                if (!firstLogged)
                {
                    Debug.Log($"{shotName} 諛쒖궗�� (Yaw: {yaw}, Pitch: {pitch}, Force: {force}) �� {sc.name}");
                    firstLogged = true;
                }
            }
        }
    }


    //********** 異붽� *********** �꾩뿉 �ㅼ쐷 �⑥닔�ㅼ쓣 �꾨옒 �⑥닔�ㅻ줈 �ъ슜�섏떆��
    // �ㅻ쾭 �ㅼ쐷: Clear, Drop
    public void Clear()
    {
        var targets = overZone != null ? new List<Shuttlecock>(overZone.GetShuttlecocks()) : new List<Shuttlecock>();
        if (targets.Count == 0) return;
        LaunchToAll(0f, 45f, 35f, "�대━��", targets);
    }

    public void Drop()
    {
        var targets = overZone != null ? new List<Shuttlecock>(overZone.GetShuttlecocks()) : new List<Shuttlecock>();
        if (targets.Count == 0) return;
        LaunchToAll(0f, 50f, 15f, "�쒕∼", targets);
    }

    // �몃뜑 �ㅼ쐷: Hairpin, Drive, Under
    public void Hairpin()
    {
        var targets = underZone != null ? new List<Shuttlecock>(underZone.GetShuttlecocks()) : new List<Shuttlecock>();
        if (targets.Count == 0) return;
        LaunchToAll(0f, 35f, 13f, "�ㅼ뼱��", targets);
    }

    public void Drive()
    {
        var targets = underZone != null ? new List<Shuttlecock>(underZone.GetShuttlecocks()) : new List<Shuttlecock>();
        if (targets.Count == 0) return;
        LaunchToAll(0f, 10f, 25f, "�쒕씪�대툕", targets);
    }

    public void Under()
    {
        var targets = underZone != null ? new List<Shuttlecock>(underZone.GetShuttlecocks()) : new List<Shuttlecock>();
        if (targets.Count == 0) return;
        LaunchToAll(0f, 30f, 20f, "�몃뜑", targets);
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

        LaunchToAll(0f, -5f, 30f, "�ㅻℓ��!!!!!!!!!!!!", targets);
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