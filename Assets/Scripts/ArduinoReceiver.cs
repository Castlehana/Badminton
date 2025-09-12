using System.IO.Ports;
using System.Threading;
using UnityEngine;

public class ArduinoReceiver : MonoBehaviour
{
    public PlayerShooting shooter;
    public PlayerMovement playerMovement;

    SerialPort sp = new SerialPort("COM5", 115200);
    Thread readThread;
    volatile bool keepReading = true;

    string latestSwingData = "";
    Vector2 latestMoveInput = Vector2.zero;
    bool jumpRequested = false;

    void Start()
    {
        sp.ReadTimeout = 100;
        try
        {
            sp.Open();
            readThread = new Thread(ReadSerial);
            readThread.Start();
            Debug.Log("Serial port opened successfully.");
        }
        catch (System.Exception e)
        {
            Debug.LogError("시리얼 포트 열기 실패: " + e.Message);
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
        // 스윙 처리
        if (!string.IsNullOrEmpty(latestSwingData))
        {
            string msg = latestSwingData;
            latestSwingData = ""; // 다음 입력을 위해 비움

            //Debug.Log($"스윙 동작 감지: {msg}");

            // "SWING:" 뒤 부분만 추출
            string command = msg.Substring(6).Trim().ToUpper();

            if (shooter != null)
            {
                switch (command)
                {
                    case "CLEAR":
                        shooter.Clear();    // 스윙은 CLEAR만 실행
                        break;

                        // 나머지 스윙 동작은 주석 처리
                        /*
                        case "DROP":
                            shooter.Drop();
                            break;
                        case "SMASH":
                            shooter.Smash();
                            break;
                        case "PUSH":
                            shooter.Push();
                            break;
                        case "HAIRPIN":
                            shooter.Hairpin();
                            break;
                        case "DRIVE":
                            shooter.Drive();
                            break;
                        default:
                            Debug.Log($"알 수 없는 SWING 명령: {command}");
                            break;
                        */
                }
            }
        }

        // 점프 처리
        if (jumpRequested && playerMovement != null)
        {
            playerMovement.Jump();   // JUMP는 그대로 실행
            jumpRequested = false;
        }

        // 이동 처리
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
        if (sp.IsOpen)
        {
            sp.Close();
        }
    }
}