using System.IO.Ports;
using System.Threading;
using UnityEngine;

public class ArduinoReceiver : MonoBehaviour
{
    // 인스펙터에서 COM 포트를 직접 입력할 수 있는 필드
    public string comPort;

    public PlayerShooting shooter;
    public PlayerMovement playerMovement;

    SerialPort sp;
    Thread readThread;
    volatile bool keepReading = true;

    string latestSwingData = "";
    Vector2 latestMoveInput = Vector2.zero;
    bool jumpRequested = false;

    private const int BAUD_RATE = 115200;

    void Start()
    {
        // 인스펙터에 입력된 COM 포트 번호로 시리얼 포트 객체 생성
        sp = new SerialPort(comPort, BAUD_RATE);
        sp.ReadTimeout = 2000;

        try
        {
            sp.Open();
            readThread = new Thread(ReadSerial);
            readThread.Start();
            Debug.Log($"시리얼 포트 {comPort}에 성공적으로 연결했습니다.");
        }
        catch (System.Exception e)
        {
            Debug.LogError($"시리얼 포트 {comPort} 열기 실패: " + e.Message);
            this.enabled = false;
        }

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

                if (line.StartsWith("SWING:"))
                {
                    latestSwingData = line;
                }
                else if (line.StartsWith("MOVE:"))
                {
                    string[] parts = line.Substring(5).Split(',');
                    if (parts.Length == 2)
                    {
                        if (int.TryParse(parts[0], out int x) && int.TryParse(parts[1], out int y))
                        {
                            latestMoveInput = new Vector2(x, y);
                        }
                    }
                }
                else if (line == "JUMP")
                {
                    jumpRequested = true;
                }
            }
            catch (System.Exception) { }
        }
    }

    void Update()
    {
        if (!string.IsNullOrEmpty(latestSwingData))
        {
            string msg = latestSwingData;
            latestSwingData = "";

            string command = msg.Substring(6).Trim().ToUpper();

            if (shooter != null)
            {
                switch (command)
                {
                    case "CLEAR":
                        shooter.Clear();
                        break;
                }
            }
        }

        if (jumpRequested && playerMovement != null)
        {
            jumpRequested = false;
        }

        if (playerMovement != null)
        {
            float normalizedX = latestMoveInput.x / 127f;
            float normalizedY = latestMoveInput.y / 127f;

            float threshold = 0.1f;
            if (Mathf.Abs(normalizedX) < threshold && Mathf.Abs(normalizedY) < threshold)
                playerMovement.SetMoveInput(Vector2.zero);
            else
                playerMovement.SetMoveInput(new Vector2(normalizedX, normalizedY));
        }
    }

    void OnApplicationQuit()
    {
        keepReading = false;
        if (readThread != null && readThread.IsAlive)
        {
            readThread.Join();
        }
        if (sp != null && sp.IsOpen)
        {
            sp.Close();
        }
    }
}