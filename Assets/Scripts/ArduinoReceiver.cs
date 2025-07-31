
using System.IO.Ports;
using System.Threading;
using UnityEngine;

public class ArduinoReceiver : MonoBehaviour
{
    public PlayerShooting shooter;
    public PlayerMovement playerMovement;

    SerialPort sp = new SerialPort("COM3", 115200); // 포트는 환경에 맞게 변경
    Thread readThread;
    volatile bool keepReading = true;

    string latestData = "";
    int joyX = 0;
    int joyY = 0;
    bool jumpRequested = false;

    void Start()
    {
        sp.ReadTimeout = 100;
        try
        {
            sp.Open();
            readThread = new Thread(ReadSerial);
            readThread.Start();
        }
        catch (System.Exception e)
        {
            Debug.LogError("시리얼 포트 열기 실패: " + e.Message);
        }

        // 자동 참조 연결 (옵션)
        if (playerMovement == null)
            playerMovement = FindObjectOfType<PlayerMovement>();
        if (shooter == null)
            shooter = FindObjectOfType<PlayerShooting>();
    }

    void ReadSerial()
    {
        while (keepReading)
        {
            try
            {
                string line = sp.ReadLine().Trim();

                if (line == "SWING")
                {
                    latestData = "SWING";
                }
                else if (line == "JUMP")
                {
                    jumpRequested = true;
                }
                else if (line.StartsWith("JOY:"))
                {
                    string[] parts = line.Substring(4).Split(',');
                    if (parts.Length == 2)
                    {
                        int.TryParse(parts[0], out joyX);
                        int.TryParse(parts[1], out joyY);
                    }
                }
            }
            catch { }
        }
    }

    void Update()
    {
        // SWING 동작 → 공격 처리
        if (latestData == "SWING")
        {
            Debug.Log("스윙 동작 감지 ✅");
            shooter?.Clear();
            latestData = "";
        }

        // C 버튼 → 점프 요청
        if (jumpRequested && playerMovement != null)
        {
            playerMovement.Jump();
            jumpRequested = false;
        }

        // 조이스틱 이동 처리
        if (playerMovement != null)
        {
            int filteredX = Mathf.Abs(joyX) < 4 ? 0 : joyX;
            int filteredY = Mathf.Abs(joyY) < 4 ? 0 : joyY;

            float h = Mathf.Clamp(filteredX / 64f, -1f, 1f);
            float v = Mathf.Clamp(filteredY / 64f, -1f, 1f);

            playerMovement.SetMoveInput(new Vector2(h, v));
        }
    }

    void OnApplicationQuit()
    {
        keepReading = false;
        if (readThread != null && readThread.IsAlive) readThread.Join();
        if (sp.IsOpen) sp.Close();
    }
}
