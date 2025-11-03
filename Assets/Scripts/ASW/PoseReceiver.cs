using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Collections.Generic;
using UnityEngine;

// Python 송신 페이로드와 동일
// { "swing": true, "type": 0..4, "label":"Clear|Drive|Drop|Under|Hairpin", "conf":0~1, "strength":float }
[Serializable]
public class SwingMsg
{
    public bool swing;
    public float strength;
    public int type;        // 0:Clear, 1:Drive, 2:Drop, 3:Under, 4:Hairpin
    public string label;    // 참고용
    public float conf;      // softmax 확률
}

public enum SwingClass { Clear = 0, Drive = 1, Drop = 2, Under = 3, Hairpin = 4 }

public class PoseReceiver : MonoBehaviour
{
    [Header("UDP")]
    public int port = 5052;

    [Header("Decision Thresholds")]
    [Range(0f, 1f)] public float minConfidence = 0.45f; // Python TH_SOFT와 일치(필요 시 0.60으로 상향)
    public float weakSpeed = 3f;                         // 속도 약 임계
    public float strongSpeed = 6f;                       // 속도 강 임계

    [Header("Latest Swing State (Read-Only)")]
    public bool swingDetected;       // 새 메시지에서 스윙 감지됨?
    public float swingSpeed;         // 확정 순간의 속도
    public int typeId;               // 0..4 (Clear,Drive,Drop,Under,Hairpin)
    public string typeLabel;         // 문자열 라벨
    public float confidence;         // softmax conf

    [Header("References")]
    public PlayerShooting playerShooting; // 인스펙터 연결

    // 내부
    private UdpClient udp;
    private IPEndPoint anyEndPoint;
    private readonly Queue<SwingMsg> inbox = new Queue<SwingMsg>(); // 메인스레드 처리용 버퍼
    private readonly object locker = new object();

    void Start()
    {
        udp = new UdpClient(port);
        anyEndPoint = new IPEndPoint(IPAddress.Any, port);
        udp.BeginReceive(ReceiveData, null);
    }

    // 비-메인 스레드 콜백 → 큐에 적재
    void ReceiveData(IAsyncResult ar)
    {
        try
        {
            byte[] data = udp.EndReceive(ar, ref anyEndPoint);
            string message = Encoding.UTF8.GetString(data);

            SwingMsg msg = null;
            try
            {
                msg = JsonUtility.FromJson<SwingMsg>(message);
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[PoseReceiver] JSON parse failed: {ex.Message}\nRaw: {message}");
            }

            if (msg != null && msg.swing)
            {
                lock (locker) inbox.Enqueue(msg);
            }
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
        // 수신된 이벤트를 메인 스레드에서 처리
        while (true)
        {
            SwingMsg msg = null;
            lock (locker)
            {
                if (inbox.Count > 0) msg = inbox.Dequeue();
            }
            if (msg == null) break;

            // 신뢰도 필터
            if (msg.conf < minConfidence) continue;

            // 최신 상태 갱신
            typeId = Mathf.Clamp(msg.type, 0, 4);
            typeLabel = string.IsNullOrEmpty(msg.label) ? ((SwingClass)typeId).ToString() : msg.label;
            confidence = msg.conf;
            swingSpeed = msg.strength;

            swingDetected = true;
            HandleSwingByClass();   // 클래스→함수 디스패치
            swingDetected = false;  // 중복 방지
        }
    }

    // 모델 5클래스 → PlayerShooting 함수 매핑
    // 0=Clear → Clear(), 1=Drive → Drive(), 2=Drop → Drop(),
    // 3=Under → UnderStrong/UnderWeak(속도 분기), 4=Hairpin → Hairpin()
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
                if (swingSpeed >= strongSpeed)
                {
                    Debug.Log($"➡ UnderStrong (speed={swingSpeed:F2}, conf={confidence:F2})");
                    playerShooting.UnderSwing();
                }
                else if (swingSpeed >= weakSpeed)
                {
                    Debug.Log($"➡ UnderWeak (speed={swingSpeed:F2}, conf={confidence:F2})");
                    playerShooting.Hairpin();
                }
                else
                {
                    Debug.Log($"➡ Under(too weak) skip (speed={swingSpeed:F2})");
                }
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
