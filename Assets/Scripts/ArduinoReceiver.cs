using System.IO.Ports;
using System.Threading;
using UnityEngine;

public class ArduinoReceiver : MonoBehaviour
{
    public PlayerShooting shooter;
    public PlayerMovement playerMovement;

    SerialPort sp = new SerialPort("COM3", 115200);
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
        if (!string.IsNullOrEmpty(latestSwingData))
        {
            Debug.Log($"스윙 동작 감지: {latestSwingData}");
            latestSwingData = "";
        }

        if (jumpRequested && playerMovement != null)
        {
            playerMovement.Jump();
            jumpRequested = false;
        }

        if (playerMovement != null)
        {
            // 아두이노에서 받은 조이스틱 값을 -127~127 범위에서 -1~1 범위로 정규화
            float normalizedX = latestMoveInput.x / 127f;
            float normalizedY = latestMoveInput.y / 127f;

            // ⚠️ 여기서 Y축 방향이 반전될 수 있습니다. 닌텐도 눈차크는 종종 Y축이 반대입니다.
            // 만약 캐릭터가 조이스틱을 앞으로 밀었을 때 뒤로 간다면, 아래 코드를 사용하세요.
            // float normalizedY = -latestMoveInput.y / 127f;

            // 정규화된 값을 PlayerMovement에 전달하기 전에 충분히 큰 값인지 확인
            // 캐릭터가 움직이기 위한 최소 입력 값 (예: 0.1f)
            float threshold = 0.1f;
            if (Mathf.Abs(normalizedX) < threshold && Mathf.Abs(normalizedY) < threshold)
            {
                playerMovement.SetMoveInput(Vector2.zero);
            }
            else
            {
                playerMovement.SetMoveInput(new Vector2(normalizedX, normalizedY));
            }
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