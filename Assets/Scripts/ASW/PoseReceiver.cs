using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Collections.Generic;
using UnityEngine;

[Serializable]
public class SwingMsgPy
{
    public bool swing;
    public string @class;   // Python JSON key "class"
    public float conf;
    public float ts;        // 미사용이지만 보관
}

[Serializable]
public class JumpMsgPy
{
    public bool jump;
    public float speed;     // -dy/dt (1/s)
    public float ts;        // 미사용이지만 보관
}

public enum SwingClass { Clear = 0, Drive = 1, Drop = 2, Under = 3, Hairpin = 4 }

public class PoseReceiver : MonoBehaviour
{
    [Header("UDP")]
    public int port = 5052;

    [Header("Decision Threshold")]
    [Range(0f, 1f)] public float minConfidence = 0.45f;

    [Header("Latest Swing State (Read-Only)")]
    public bool swingDetected;
    public int typeId;           // 0..4
    public string typeLabel;     // "Clear" 등
    public float confidence;     // softmax 확률

    [Header("Jump (Read-Only)")]
    public bool jumpDetected;
    public float lastJumpSpeed;  // Python speed 값(-dy/dt)

    [Header("References")]
    public PlayerShooting playerShooting; // 인스펙터에서 연결
    public AutoMovement autoMovement;     // 인스펙터에서 연결 → Jump() 호출

    private UdpClient udp;
    private IPEndPoint anyEndPoint;

    // 메인 스레드 처리를 위한 큐
    private readonly Queue<SwingMsgPy> swingInbox = new Queue<SwingMsgPy>();
    private readonly Queue<JumpMsgPy> jumpInbox = new Queue<JumpMsgPy>();
    private readonly object locker = new object();

    void Start()
    {
        udp = new UdpClient(port);
        anyEndPoint = new IPEndPoint(IPAddress.Any, port);
        udp.BeginReceive(ReceiveData, null);
        Debug.Log($"[PoseReceiver] Listening UDP :{port} (Python swing & jump schema)");
    }

    // 문자열 라벨 → enum 인덱스
    private static int LabelToTypeId(string label)
    {
        if (string.IsNullOrEmpty(label)) return -1;
        switch (label.ToLowerInvariant())
        {
            case "clear": return (int)SwingClass.Clear;
            case "drive": return (int)SwingClass.Drive;
            case "drop": return (int)SwingClass.Drop;
            case "under": return (int)SwingClass.Under;
            case "hairpin": return (int)SwingClass.Hairpin;
            case "ready": return -1; // 스윙 아님
            default: return -1;
        }
    }

    // 비-메인 스레드 콜백 → 큐 적재
    void ReceiveData(IAsyncResult ar)
    {
        try
        {
            byte[] data = udp.EndReceive(ar, ref anyEndPoint);
            string message = Encoding.UTF8.GetString(data);

            // 1) 스윙 패킷 시도: {"swing":true,"class":"Clear","conf":0.98,"ts":...}
            try
            {
                var sw = JsonUtility.FromJson<SwingMsgPy>(message);
                if (sw != null)
                {
                    int tid = LabelToTypeId(sw.@class);
                    if (sw.swing && tid >= 0)
                    {
                        lock (locker) swingInbox.Enqueue(sw);
                        // 스윙으로 파싱됐으면 종료
                        udp.BeginReceive(ReceiveData, null);
                        return;
                    }
                }
            }
            catch { /* fallthrough to jump */ }

            // 2) 점프 패킷 시도: {"jump":true,"speed":2.43,"ts":...}
            try
            {
                var jp = JsonUtility.FromJson<JumpMsgPy>(message);
                if (jp != null && jp.jump)
                {
                    lock (locker) jumpInbox.Enqueue(jp);
                    udp.BeginReceive(ReceiveData, null);
                    return;
                }
            }
            catch { /* ignore */ }
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[PoseReceiver] Receive error: {e.Message}");
        }
        finally
        {
            try { udp.BeginReceive(ReceiveData, null); } catch { }
        }
    }

    void OnDestroy()
    {
        try { udp?.Close(); } catch { }
    }

    void Update()
    {
        // ---------- 스윙 처리 ----------
        while (true)
        {
            SwingMsgPy msg = null;
            lock (locker) { if (swingInbox.Count > 0) msg = swingInbox.Dequeue(); }
            if (msg == null) break;

            if (msg.conf < minConfidence) continue;

            typeId = LabelToTypeId(msg.@class);
            if (typeId < 0) continue;

            typeLabel = msg.@class;
            confidence = msg.conf;

            swingDetected = true;
            HandleSwingByClass();
            swingDetected = false;
        }

        // ---------- 점프 처리 ----------
        while (true)
        {
            JumpMsgPy jmsg = null;
            lock (locker) { if (jumpInbox.Count > 0) jmsg = jumpInbox.Dequeue(); }
            if (jmsg == null) break;

            lastJumpSpeed = jmsg.speed;
            jumpDetected = true;

            if (autoMovement != null)
            {
                try { autoMovement.Jump(); }
                catch (Exception ex) { Debug.LogWarning($"[PoseReceiver] autoMovement.Jump() error: {ex.Message}"); }
            }
            else
            {
                Debug.Log($"[PoseReceiver] Jump received (speed={lastJumpSpeed:F3}) but AutoMovement not assigned.");
            }

            // 한 프레임 내 중복 호출 방지용 플래그 리셋
            jumpDetected = false;
        }
    }

    private void HandleSwingByClass()
    {
        if (playerShooting == null) return;

        switch (typeId)
        {
            case (int)SwingClass.Clear:
                Debug.Log($"➡ Clear (conf={confidence:F2})");
                playerShooting.ClearSwing();
                break;

            case (int)SwingClass.Drive:
                Debug.Log($"➡ Drive (conf={confidence:F2})");
                playerShooting.DriveSwing();
                break;

            case (int)SwingClass.Drop:
                Debug.Log($"➡ Drop (conf={confidence:F2})");
                playerShooting.DropSwing();
                break;

            case (int)SwingClass.Under:
                Debug.Log($"➡ Under (conf={confidence:F2})");
                playerShooting.UnderSwing();
                break;

            case (int)SwingClass.Hairpin:
                Debug.Log($"➡ Hairpin (conf={confidence:F2})");
                playerShooting.Hairpin();
                break;

            default:
                Debug.LogWarning($"[PoseReceiver] Unknown class id: {typeId}");
                break;
        }
    }
}
