using System;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using UnityEngine;
using InTheHand.Bluetooth;

public class ESP32BleReceiver : MonoBehaviour
{
    [Header("BLE Filter")]
    [Tooltip("스캔 후 이름 일치로 선택합니다. (플랫폼에 따라 이름이 null일 수 있음)")]
    public string deviceName = "ESP32_Controller";

    [Tooltip("Nordic UART Service UUID (NUS)")]
    public string serviceUuid = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E";

    [Tooltip("TX(ESP->App) Notify Characteristic UUID")]
    public string txCharUuid = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E";

    [Tooltip("RX(App->ESP) Write Characteristic UUID (옵션)")]
    public string rxCharUuid = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E";

    [Header("Lifecycle")]
    public bool autoConnectOnEnable = true;

    [Header("Debug")]
    public bool logPackets = true;
    public int logPacketsMax = 10;

    // 상태
    public bool IsConnected { get; private set; }
    public string ConnectedDeviceName { get; private set; } = "";
    public double LastReceivedAt { get; private set; } = -1;
    public long PacketCount => Interlocked.Read(ref _packetCount);

    private readonly ConcurrentQueue<string> _msgQueue = new ConcurrentQueue<string>();
    private long _packetCount = 0;
    private long _tickCounter = 0;
    private readonly StringBuilder _lineBuf = new StringBuilder();
    private int _logged = 0;

    // InTheHand 객체
    private BluetoothDevice _device;
    private GattCharacteristic _txChar;
    private GattCharacteristic _rxChar;
    private CancellationTokenSource _cts;

    // 추가: 재연결 쿨다운 변수
    private const float RECONNECT_COOLDOWN = 5f; // 5초마다 재연결 시도
    private float _lastReconnectTime = -RECONNECT_COOLDOWN;

    void OnEnable()
    {
        if (autoConnectOnEnable)
            _ = StartClientAsync();
    }

    void OnDisable()
    {
        _ = StopClientAsync();
    }

    void Update()
    {
        while (Interlocked.Read(ref _tickCounter) > 0)
        {
            LastReceivedAt = Time.realtimeSinceStartupAsDouble;
            Interlocked.Decrement(ref _tickCounter);
        }

        // 연결이 끊겼을 때 재연결 로직 추가
        if (!IsConnected && Time.time - _lastReconnectTime > RECONNECT_COOLDOWN)
        {
            _lastReconnectTime = Time.time;
            _ = StartClientAsync();
        }
    }

    /// <summary>수신 큐에서 한 줄 꺼내기</summary>
    public string DequeueMessage() => _msgQueue.TryDequeue(out var s) ? s : null;

    /// <summary>최근 windowSec 동안 알림이 있었는지</summary>
    public bool IsAlive(float windowSec)
    {
        if (LastReceivedAt < 0) return false;
        return (Time.realtimeSinceStartupAsDouble - LastReceivedAt) <= windowSec;
    }

    /// <summary>ESP32에 문자열 한 줄 쓰기 (옵션)</summary>
    public async Task SendAsync(string line)
    {
        if (_rxChar == null || !IsConnected) return;
        var data = Encoding.UTF8.GetBytes(line);
        try { await _rxChar.WriteValueWithoutResponseAsync(data); }
        catch (Exception ex) { Debug.LogWarning("[BLE] SendAsync error: " + ex.Message); }
    }

    public async Task StartClientAsync()
    {
        await StopClientAsync();
        _cts = new CancellationTokenSource();

        Debug.Log("[BLE] Scanning for device...");

        BluetoothDevice device = null;
        try
        {
            var opts = new RequestDeviceOptions
            {
                // 일부 플랫폼은 필터 무시 → AcceptAll 이후 수동 필터
                AcceptAllDevices = true
            };

            var devices = await Bluetooth.ScanForDevicesAsync(opts);
            foreach (var d in devices)
            {
                var name = d.Name;
                if (!string.IsNullOrEmpty(name) && name.Equals(deviceName, StringComparison.Ordinal))
                {
                    device = d;
                    break;
                }
            }
        }
        catch (Exception ex)
        {
            Debug.LogError("[BLE] Scan error: " + ex);
        }

        if (device == null)
        {
            Debug.LogWarning("[BLE] Target device not found");
            return;
        }

        _device = device;
        ConnectedDeviceName = _device.Name ?? "(no-name)";

        try
        {
            // ✅ 이 빌드에서 ConnectAsync는 반환값 없는 Task
            await _device.Gatt.ConnectAsync();

            // ✅ GATT 인스턴스는 _device.Gatt 로 바로 사용
            var service = await _device.Gatt.GetPrimaryServiceAsync(Guid.Parse(serviceUuid));

            // ✅ 어떤 빌드든 동작하도록 characteristic 안전 조회
            var allChars = await GetAllCharacteristicsCompatAsync(service);

            _txChar = allChars.FirstOrDefault(c => c.Uuid == Guid.Parse(txCharUuid));
            if (_txChar == null)
                throw new Exception("TX characteristic not found: " + txCharUuid);

            _rxChar = allChars.FirstOrDefault(c => c.Uuid == Guid.Parse(rxCharUuid)); // 없어도 OK

            // InTheHand v4 이벤트명
            _txChar.CharacteristicValueChanged += OnTxValueChanged;
            await _txChar.StartNotificationsAsync();

            IsConnected = true;
            Debug.Log($"[BLE] Connected to \"{ConnectedDeviceName}\" & subscribed.");
        }
        catch (Exception ex)
        {
            Debug.LogError("[BLE] Connect error: " + ex);
            IsConnected = false;
        }
    }

    public async Task StopClientAsync()
    {
        try
        {
            if (_txChar != null)
            {
                try { await _txChar.StopNotificationsAsync(); } catch { /* ignore */ }
                _txChar.CharacteristicValueChanged -= OnTxValueChanged;
                _txChar = null;
            }

            _rxChar = null;

            _device?.Dispose();
            _device = null;
        }
        catch { /* ignore */ }

        _cts?.Cancel();
        _cts = null;

        IsConnected = false;
    }

    // 수신 처리: e.Value는 byte[] (현재 빌드 기준)
    private void OnTxValueChanged(object sender, GattCharacteristicValueChangedEventArgs e)
    {
        try
        {
            var bytes = e.Value; // byte[]
            if (bytes == null || bytes.Length == 0) return;

            string chunk = Encoding.UTF8.GetString(bytes);

            lock (_lineBuf)
            {
                _lineBuf.Append(chunk);
                var s = _lineBuf.ToString();
                int idx;
                while ((idx = s.IndexOf('\n')) >= 0)
                {
                    var line = s.Substring(0, idx).TrimEnd('\r');
                    s = s.Substring(idx + 1);

                    if (!string.IsNullOrEmpty(line))
                    {
                        _msgQueue.Enqueue(line);
                        Interlocked.Increment(ref _packetCount);
                        Interlocked.Increment(ref _tickCounter);

                        if (logPackets && _logged < logPacketsMax)
                        {
                            Debug.Log($"[BLE<-ESP32] \"{line}\"");
                            _logged++;
                        }
                    }
                }

                _lineBuf.Length = 0;
                _lineBuf.Append(s);
            }
        }
        catch (Exception ex)
        {
            Debug.LogWarning("[BLE] CharacteristicValueChanged error: " + ex.Message);
        }
    }

    // ===== 호환 헬퍼: InTheHand 빌드별 characteristic 접근 방식 커버 =====
    private async Task<IReadOnlyList<GattCharacteristic>> GetAllCharacteristicsCompatAsync(GattService service)
    {
        // 1) 먼저 비동기 초기화 메서드 호출(반환값 void인 빌드 대응)
        try
        {
            var mInit = service.GetType().GetMethod(
                "GetCharacteristicsAsync",
                BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic,
                null, Type.EmptyTypes, null);

            if (mInit != null)
            {
                var taskObj = mInit.Invoke(service, null) as Task;
                if (taskObj != null) await taskObj; // void Task
            }
        }
        catch { /* 일부 빌드엔 없음 */ }

        // 2) property "Characteristics" 시도
        try
        {
            var p = service.GetType().GetProperty(
                "Characteristics",
                BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);

            if (p != null)
            {
                var val = p.GetValue(service, null) as IEnumerable<GattCharacteristic>;
                if (val != null) return new List<GattCharacteristic>(val);
            }
        }
        catch { }

        // 3) method "GetCharacteristics()" 시도
        try
        {
            var mList = service.GetType().GetMethod(
                "GetCharacteristics",
                BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic,
                null, Type.EmptyTypes, null);

            if (mList != null)
            {
                var val = mList.Invoke(service, null) as IEnumerable<GattCharacteristic>;
                if (val != null) return new List<GattCharacteristic>(val);
            }
        }
        catch { }

        // 4) 최후 수단: "GetCharacteristicAsync(Guid)"가 있으면 필요한 UUID만 직접 로드
        var list = new List<GattCharacteristic>();
        try
        {
            var mOne = service.GetType().GetMethod(
                "GetCharacteristicAsync",
                BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic,
                null, new[] { typeof(Guid) }, null);

            if (mOne != null)
            {
                // TX
                var txTaskObj = mOne.Invoke(service, new object[] { Guid.Parse(txCharUuid) });
                var txChar = await AwaitTaskWithResult<GattCharacteristic>(txTaskObj);
                if (txChar != null) list.Add(txChar);

                // RX (옵션)
                try
                {
                    var rxTaskObj = mOne.Invoke(service, new object[] { Guid.Parse(rxCharUuid) });
                    var rxChar = await AwaitTaskWithResult<GattCharacteristic>(rxTaskObj);
                    if (rxChar != null) list.Add(rxChar);
                }
                catch { /* RX 없을 수 있음 */ }
            }
        }
        catch { }

        return list;
    }

    // Task<T>를 object로 받아서 안전하게 await + Result 꺼내기
    private static async Task<T> AwaitTaskWithResult<T>(object taskObj) where T : class
    {
        if (taskObj is Task t)
        {
            await t;
            var type = taskObj.GetType();
            var prop = type.GetProperty("Result", BindingFlags.Instance | BindingFlags.Public);
            if (prop != null)
            {
                var val = prop.GetValue(taskObj);
                return val as T;
            }
        }
        return null;
    }
}
