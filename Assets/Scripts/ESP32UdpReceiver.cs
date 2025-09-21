using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Collections.Concurrent;
using UnityEngine;
using Debug = UnityEngine.Debug;

public class ESP32UdpReceiver : MonoBehaviour
{
    [Header("Network")]
    public int listenPort = 42100;
    public bool allowBroadcast = true;

    [Header("Debug")]
    public bool logPackets = true;
    public int logPacketsMax = 10;

    [Header("Lifecycle")]
    public bool autoStart = true;

    // --- Internals ---
    private UdpClient _client;
    private CancellationTokenSource _cts;
    private Task _recvTask;

    private readonly ConcurrentQueue<string> _msgQueue = new ConcurrentQueue<string>();

    // --- Link status (for HUD) ---
    // PacketCount는 필드로 관리하고, 공개 프로퍼티는 읽기 전용 게터만 둡니다.
    private long _packetCount = 0;
    public long PacketCount => Interlocked.Read(ref _packetCount);

    public double LastReceivedAt { get; private set; } = -1;
    public string LastFrom { get; private set; } = "";
    public string LastMsg { get; private set; } = "";

    public bool IsAlive(float windowSec)
    {
        if (LastReceivedAt < 0) return false;
        return (Time.realtimeSinceStartupAsDouble - LastReceivedAt) <= windowSec;
    }

    // 메인스레드에서만 LastReceivedAt 갱신하기 위한 신호 카운터(필드이므로 ref 사용 OK)
    private long _tickCounter = 0;

    void OnEnable()
    {
        if (autoStart) StartListener();
    }

    void OnDisable()
    {
        StopListener();
    }

    void Update()
    {
        while (Interlocked.Read(ref _tickCounter) > 0)
        {
            LastReceivedAt = Time.realtimeSinceStartupAsDouble;
            Interlocked.Decrement(ref _tickCounter);
        }
    }

    public string DequeueMessage() => _msgQueue.TryDequeue(out var msg) ? msg : null;

    public void StartListener()
    {
        if (_client != null) return;

        try
        {
            _client = new UdpClient(listenPort);
            _client.EnableBroadcast = allowBroadcast;
        }
        catch (Exception e)
        {
            Debug.LogError($"[ESP32UDP] 소켓 초기화 실패: {e}");
            enabled = false;
            return;
        }

        _cts = new CancellationTokenSource();
        _recvTask = Task.Run(() => ReceiveLoopAsync(_cts.Token), _cts.Token);
        Debug.Log($"[ESP32UDP] Listening on UDP *:{listenPort} (broadcast={allowBroadcast})");
    }

    public void StopListener()
    {
        try { _cts?.Cancel(); } catch { }
        try { _client?.Close(); } catch { }
        try { _client?.Dispose(); } catch { }
        _client = null;

        try { _recvTask?.Wait(200); } catch { }
        _recvTask = null;

        try { _cts?.Dispose(); } catch { }
        _cts = null;
    }

    private async Task ReceiveLoopAsync(CancellationToken token)
    {
        int logged = 0;

        while (!token.IsCancellationRequested)
        {
            UdpReceiveResult result;
            try
            {
                // Unity(.NET Standard 2.1) 호환: WaitAsync 사용 안 함
                result = await _client.ReceiveAsync();
            }
            catch (ObjectDisposedException)
            {
                break; // 소켓 닫힘 = 정상 종료
            }
            catch (Exception ex)
            {
                // 취소 중이면 빠져나감
                if (token.IsCancellationRequested) break;
                Debug.LogWarning($"[ESP32UDP] Receive 예외: {ex.Message}");
                continue;
            }

            if (result.Buffer == null || result.Buffer.Length == 0) continue;

            string msg = Encoding.UTF8.GetString(result.Buffer);

            // 스레드 세이프 멤버만 직접 갱신
            LastFrom = result.RemoteEndPoint.Address.ToString();
            LastMsg = msg;
            Interlocked.Increment(ref _packetCount);   // ✅ 필드에만 ref 사용
            Interlocked.Increment(ref _tickCounter);   // ✅ 필드에만 ref 사용

            _msgQueue.Enqueue(msg);

            if (logPackets && logged < logPacketsMax)
            {
                Debug.Log($"[ESP32UDP] {LastFrom}:{result.RemoteEndPoint.Port} -> \"{msg}\"");
                logged++;
            }
        }
    }
}
