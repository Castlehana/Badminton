using UnityEngine;

public class ESP32LinkStatusHUD : MonoBehaviour
{
    [Header("References")]
    public ESP32UdpReceiver receiver;

    [Header("Display")]
    public float aliveWindowSec = 2.5f;     // 최근 몇 초 이내 수신이면 연결 OK로 표시
    public Vector2 margin = new Vector2(12, 12);
    public Vector2 boxSize = new Vector2(360, 96);
    public bool showLastMessage = true;

    void OnGUI()
    {
        if (receiver == null) return;

        bool alive = receiver.IsAlive(aliveWindowSec);
        string status = alive ? "CONNECTED" : "NO SIGNAL";
        double age = (receiver.LastReceivedAt > 0)
            ? (Time.realtimeSinceStartupAsDouble - receiver.LastReceivedAt)
            : double.PositiveInfinity;

        // 우상단 위치 계산
        float x = Screen.width - boxSize.x - margin.x;
        float y = margin.y;

        // 배경 박스
        var oldColor = GUI.color;
        GUI.color = alive ? new Color(0.2f, 0.8f, 0.2f, 0.85f) : new Color(0.9f, 0.3f, 0.3f, 0.85f);
        GUI.Box(new Rect(x, y, boxSize.x, boxSize.y), GUIContent.none);
        GUI.color = Color.white;

        // 텍스트
        var labelRect = new Rect(x + 10, y + 8, boxSize.x - 20, boxSize.y - 16);
        string body =
            $"ESP32 Link: {status}\n" +
            $"Packets: {receiver.PacketCount}   Last: {(float)age:0.0}s ago\n" +
            $"From: {receiver.LastFrom}   Port: {receiver.listenPort}";
        if (showLastMessage && !string.IsNullOrEmpty(receiver.LastMsg))
            body += $"\nMsg: {receiver.LastMsg}";

        GUI.Label(labelRect, body);
        GUI.color = oldColor;
    }
}
