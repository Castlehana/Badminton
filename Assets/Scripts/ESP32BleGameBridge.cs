using System;
using UnityEngine;
using UnityEngine.UI;

public class ESP32BleGameBridge : MonoBehaviour
{
    [Header("Refs")]
    public ESP32BleReceiver ble;
    public PlayerMovement playerMovement;
    public PlayerShooting shooter;

    // UI를 직접 그리는 방식으로 변경하여, UI 필드는 더 이상 필요하지 않습니다.

    void Update()
    {
        if (ble == null)
        {
            return;
        }

        // --- Process received messages from the BLE queue ---
        string msg;
        while ((msg = ble.DequeueMessage()) != null)
        {
            // ---- SWING handling ----
            if (msg.StartsWith("SWING:"))
            {
                var cmd = msg.Substring(6).Trim().ToUpper();
                switch (cmd)
                {
                    case "CLEAR": shooter?.Clear(); break;
                    case "DROP": shooter?.Drop(); break;
                    case "SMASH": shooter?.Smash(); break;
                    case "PUSH": shooter?.Push(); break;
                    case "HAIRPIN": shooter?.Hairpin(); break;
                    case "DRIVE": shooter?.Drive(); break;
                    default:
                        Debug.Log($"[BLE] Unknown SWING: {cmd}");
                        break;
                }
            }
            // ---- MOVE handling ----
            else if (msg.StartsWith("MOVE:"))
            {
                var xy = msg.Substring(5).Split(',');
                if (xy.Length == 2 &&
                    int.TryParse(xy[0], out var mx) &&
                    int.TryParse(xy[1], out var my))
                {
                    // Normalize from 0..255 range to -1..+1
                    float nx = (mx - 127) / 128f;
                    float ny = (my - 127) / 128f;

                    playerMovement?.SendMessage(
                        "SetInput",
                        new Vector2(nx, ny),
                        SendMessageOptions.DontRequireReceiver
                    );
                }
            }
            // ---- JUMP handling ----
            else if (msg == "JUMP")
            {
                playerMovement?.SendMessage(
                    "Jump",
                    SendMessageOptions.DontRequireReceiver
                );
            }
            // ---- HELLO (initial connection) ----
            else if (msg == "HELLO")
            {
                Debug.Log("[BLE] ESP32 Connected OK");
            }
            else
            {
                Debug.Log($"[BLE] Unknown message: {msg}");
            }
        }
    }

    void OnGUI()
    {
        // 배경 박스 스타일 설정
        const float boxWidth = 200;
        const float boxHeight = 70;
        float screenWidth = Screen.width;

        GUI.Box(new Rect(screenWidth - boxWidth - 10, 10, boxWidth, boxHeight), "BLE Status");

        // 텍스트 스타일 설정 (선택사항)
        GUIStyle style = new GUIStyle(GUI.skin.label);
        style.normal.textColor = Color.white;
        style.fontSize = 18;
        style.fontStyle = FontStyle.Bold;

        // UI에 상태 정보 표시
        GUILayout.BeginArea(new Rect(screenWidth - boxWidth, 35, boxWidth - 20, boxHeight - 20));
        if (ble == null)
        {
            GUILayout.Label("Receiver not assigned!", style);
        }
        else if (ble.IsConnected)
        {
            GUILayout.Label("Connected!", style);
            GUILayout.Label($"Packets: {ble.PacketCount}", style);
        }
        else
        {
            GUILayout.Label("Disconnected...", style);
        }
        GUILayout.EndArea();
    }
}
