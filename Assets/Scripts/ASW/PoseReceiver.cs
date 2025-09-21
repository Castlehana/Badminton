using System.Net;
using System.Net.Sockets;
using System.Text;
using UnityEngine;

public class PoseReceiver : MonoBehaviour
{
    UdpClient udp;
    public int port = 5052;

    public bool swingDetected;
    public float swingSpeed;

    [Header("References")]
    public PlayerShooting playerShooting; // ✅ 인스펙터에서 연결

    void Start()
    {
        udp = new UdpClient(port);
        udp.BeginReceive(ReceiveData, null);
    }

    void ReceiveData(System.IAsyncResult ar)
    {
        IPEndPoint end = new IPEndPoint(IPAddress.Any, port);
        byte[] data = udp.EndReceive(ar, ref end);
        string message = Encoding.UTF8.GetString(data);

        // JSON에서 swing 여부 확인
        if (message.Contains("\"swing\": true"))
        {
            swingDetected = true;

            // speed 값 추출
            int idx = message.IndexOf("speed");
            if (idx > 0)
            {
                string num = message.Substring(idx).Split(':')[1].Replace("}", "").Trim();
                float.TryParse(num, out swingSpeed);
            }

            Debug.Log($"Swing detected! Speed: {swingSpeed}");
        }
        else
        {
            swingDetected = false;
        }

        udp.BeginReceive(ReceiveData, null);
    }

    void Update()
    {
        if (swingDetected && playerShooting != null)
        {
            if (swingSpeed >= 6f)
            {
                playerShooting.Clear();
                Debug.Log("➡ Clear 실행됨 (속도 6 이상)");
            }
            else if (swingSpeed >= 3f)
            {
                playerShooting.Drop();
                Debug.Log("➡ Drop 실행됨 (속도 3 이상 6 미만)");
            }

            swingDetected = false; // 중복 호출 방지
        }
    }
}
