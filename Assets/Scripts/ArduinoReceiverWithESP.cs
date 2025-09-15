using System.IO.Ports;
using System.Threading;
using UnityEngine;
using System;
using System.Text;
using System.Collections.Generic;

public class ArduinoReceiverWithESP : MonoBehaviour
{
    public PlayerShooting shooter;
    public PlayerMovement playerMovement;

    SerialPort sp = new SerialPort("COM5", 115200);
    Thread readThread;
    volatile bool keepReading = true;

    string latestSwingData = "";
    Vector2 latestMoveInput = Vector2.zero;   // raw 0~255 보관
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

    // ===== 추가: 토큰 기준으로 라인 안에 붙은 메시지 분리 =====
    static readonly string[] TOKENS = new[] { "SWING:", "MOVE:", "JUMP" };
    IEnumerable<string> ProcessPossiblyStuckLine(string line)
    {
        // 완전한 단일 메시지는 그대로 반환
        bool hasAny = false;
        foreach (var t in TOKENS) if (line.Contains(t)) { hasAny = true; break; }
        if (!hasAny) { yield return line.Trim(); yield break; }

        // 토큰의 시작 인덱스를 모두 찾아 정렬
        var starts = new List<(int idx, string key)>();
        foreach (var t in TOKENS)
        {
            int searchFrom = 0;
            while (true)
            {
                int i = line.IndexOf(t, searchFrom, StringComparison.OrdinalIgnoreCase);
                if (i < 0) break;
                starts.Add((i, t));
                searchFrom = i + t.Length;
            }
        }
        if (starts.Count == 0) { yield return line.Trim(); yield break; }

        starts.Sort((a, b) => a.idx.CompareTo(b.idx));

        // 각 토큰 구간으로 잘라 반환
        for (int k = 0; k < starts.Count; k++)
        {
            int s = starts[k].idx;
            int e = (k + 1 < starts.Count) ? starts[k + 1].idx : line.Length;
            string seg = line.Substring(s, e - s).Trim();
            if (!string.IsNullOrEmpty(seg))
                yield return seg;
        }
    }

    void ReadSerial()
    {
        while (keepReading)
        {
            try
            {
                string raw = sp.ReadLine(); // 개행 단위로 읽되,
                if (raw == null) continue;

                string line = raw.Trim();

                // 라인 안에 메시지가 붙어 올 수 있어 안전 분리
                foreach (var piece in ProcessPossiblyStuckLine(line))
                {
                    if (piece.StartsWith("SWING:", StringComparison.OrdinalIgnoreCase))
                    {
                        latestSwingData = piece; // 최신만 유지(기존 로직 그대로)
                    }
                    else if (piece.StartsWith("MOVE:", StringComparison.OrdinalIgnoreCase))
                    {
                        string body = piece.Substring(5).Trim();
                        string[] parts = body.Split(',');
                        if (parts.Length == 2)
                        {
                            if (int.TryParse(parts[0], out int x) && int.TryParse(parts[1], out int y))
                            {
                                // 0~255 범위로 가드
                                x = Mathf.Clamp(x, 0, 255);
                                y = Mathf.Clamp(y, 0, 255);
                                latestMoveInput = new Vector2(x, y);
                            }
                        }
                    }
                    else if (string.Equals(piece, "JUMP", StringComparison.OrdinalIgnoreCase))
                    {
                        jumpRequested = true;
                    }
                    // 그 외 문자열은 무시 (기존 동작과 동일)
                }
            }
            catch (System.Exception)
            {
                // 타임아웃/잡음 무시 (기존과 동일)
            }
        }
    }

    void Update()
    {
        // 스윙 처리 (기존 로직 동일)
        if (!string.IsNullOrEmpty(latestSwingData))
        {
            string msg = latestSwingData;
            latestSwingData = ""; // 다음 입력을 위해 비움

            // "SWING:" 뒤 부분만 추출
            string command = msg.Substring(6).Trim().ToUpper();

            if (shooter != null)
            {
                switch (command)
                {
                    case "CLEAR":
                        shooter.Clear();    // 스윙은 CLEAR만 실행
                        break;

                        // 확장용 주석 (기존 유지)
                        /*
                        case "DROP": shooter.Drop(); break;
                        case "SMASH": shooter.Smash(); break;
                        case "PUSH": shooter.Push(); break;
                        case "HAIRPIN": shooter.Hairpin(); break;
                        case "DRIVE": shooter.Drive(); break;
                        default: Debug.Log($"알 수 없는 SWING 명령: {command}"); break;
                        */
                }
            }
        }

        // 점프 처리 (기존 로직 동일)
        if (jumpRequested && playerMovement != null)
        {
            playerMovement.Jump();
            jumpRequested = false;
        }

        // 이동 처리 — ★ 여기만 정확한 맵핑으로 수정 ★
        if (playerMovement != null)
        {
            // latestMoveInput은 0~255, 중립 127
            float normalizedX = (latestMoveInput.x - 127f) / 127f; // -> -1..+1
            float normalizedY = (latestMoveInput.y - 127f) / 127f; // -> -1..+1

            // 소폭의 데드존 유지(기존 threshold 그대로)
            float threshold = 0.1f;
            if (Mathf.Abs(normalizedX) < threshold && Mathf.Abs(normalizedY) < threshold)
                playerMovement.SetMoveInput(Vector2.zero);
            else
                playerMovement.SetMoveInput(new Vector2(
                    Mathf.Clamp(normalizedX, -1f, 1f),
                    Mathf.Clamp(normalizedY, -1f, 1f)
                ));
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
