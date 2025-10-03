using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using UnityEngine;

[Serializable]
public class SwingMsg
{
    public bool swing;      // 스윙 발생 여부 (Python: true/false)
    public float strength;  // 스윙 확정 순간의 속도 (Python: last_swing_speed)
    public int type;        // Python 매핑: Overswing=1, Underswing=0
}

public class PoseReceiver : MonoBehaviour
{
    UdpClient udp;
    public int port = 5052;

    [Header("Latest Swing State (Read-Only)")]
    public bool swingDetected;   // 새 메시지에서 스윙 감지됨?
    public float swingSpeed;     // "확정" 순간의 속도 (Last Swing)
    public int swingType01;      // 요청 매핑: Over=0, Under=1
    public bool isUnder;         // 요청 bool: Under면 true, Over면 false

    [Header("References")]
    public PlayerShooting playerShooting; // ✅ 인스펙터에서 연결

    void Start()
    {
        udp = new UdpClient(port);
        udp.BeginReceive(ReceiveData, null);
    }

    void ReceiveData(System.IAsyncResult ar)
    {
        try
        {
            IPEndPoint end = new IPEndPoint(IPAddress.Any, port);
            byte[] data = udp.EndReceive(ar, ref end);
            string message = Encoding.UTF8.GetString(data);

            // JSON 파싱
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
                // Python -> Unity 매핑 보정
                // Python: Overswing=1, Underswing=0
                // 요청:   Over=0, Under=1  (반대로 매핑)
                int incomingType = msg.type;
                swingType01 = (incomingType == 1) ? 0 : 1;  // 1(Over)->0, 0(Under)->1
                isUnder = (swingType01 == 1);

                swingSpeed = msg.strength;
                swingDetected = true;

                Debug.Log($"[PoseReceiver] Swing detected | speed={swingSpeed:F2}, type01(Over=0,Under=1)={swingType01}, isUnder={isUnder}");
            }
            else
            {
                swingDetected = false;
            }
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[PoseReceiver] Receive error: {e.Message}");
        }
        finally
        {
            // 다음 패킷 대기
            try { udp.BeginReceive(ReceiveData, null); } catch { }
        }
    }

    void Update()
    {
        // 예시: 기존 로직은 Over 스윙(가정) 기준으로 작성되어 있으니 그대로 유지
        // 필요 시 isUnder를 활용해 언더 스윙용 분기도 넣으면 됨.
        if (swingDetected && playerShooting != null)
        {
            if (!isUnder)
            {
                // Over 스윙 처리 (요청 매핑 기준: swingType01 == 0)
                if (swingSpeed >= 6f)
                {
                    Debug.Log("➡ OverStrong 실행 (속도 ≥ 6)");
                    playerShooting.OverStrong();
                }
                else if (swingSpeed >= 3f)
                {
                    Debug.Log("➡ OverWeak 실행 (3 ≤ 속도 < 6)");
                    playerShooting.OverWeak();
                }
            }
            else
            {
                // Under 스윙 처리 (필요 시 메서드가 있다면 호출)
                // 아래는 예시. 실제로 PlayerShooting에 메서드가 없다면 주석 유지.
                
                if (swingSpeed >= 6f)
                {
                    Debug.Log("➡ UnderStrong 실행 (속도 ≥ 6)");
                    playerShooting.UnderStrong();
                }
                else if (swingSpeed >= 3f)
                {
                    Debug.Log("➡ UnderWeak 실행 (3 ≤ 속도 < 6)");
                    playerShooting.UnderWeak();
                }
                
                Debug.Log($"[PoseReceiver] Under 스윙 감지됨 (speed={swingSpeed:F2}) — 전용 처리 로직을 연결하세요.");
            }

            // 중복 호출 방지
            swingDetected = false;
        }
    }
}
