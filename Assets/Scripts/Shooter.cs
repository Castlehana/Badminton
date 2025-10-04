using System.Collections;
using UnityEngine;

public class Shooter : MonoBehaviour
{
    [Header("셔틀콕 프리팹")]
    public GameObject shuttlecockPrefab;   // Inspector에서 프리팹 지정
    [Tooltip("씬에 이 태그를 가진 오브젝트가 하나도 없을 때 1초 후 발사")]
    public string shuttlecockTag = "Shuttlecock";
    public float respawnDelay = 1f;

    [Header("XZ 방향 각도 범위 (Y축 회전)")]
    public float minYaw = 170f;
    public float maxYaw = 190f;

    [Header("위쪽 각도 범위 (Pitch)")]
    public float minPitch = 120f;
    public float maxPitch = 150f;

    [Header("발사 속도 범위")]
    public float minForce = 30f;
    public float maxForce = 60f;

    private bool _spawnScheduled = false;

    void Update()
    {
        // 씬에 'shuttlecockTag' 태그가 하나도 없고, 아직 예약되지 않았다면 1초 후 발사 예약
        if (!_spawnScheduled && GameObject.FindGameObjectWithTag(shuttlecockTag) == null)
        {
            StartCoroutine(SpawnAfterDelay());
        }
    }

    IEnumerator SpawnAfterDelay()
    {
        _spawnScheduled = true;
        yield return new WaitForSeconds(respawnDelay);

        // 대기 중에 이미 누군가 생성한 경우 스킵
        if (GameObject.FindGameObjectWithTag(shuttlecockTag) != null)
        {
            _spawnScheduled = false;
            yield break;
        }

        FireRandomShuttlecock();
        _spawnScheduled = false;
    }

    void FireRandomShuttlecock()
    {
        if (shuttlecockPrefab == null)
        {
            Debug.LogWarning("[Shooter] shuttlecockPrefab이 비어 있습니다.");
            return;
        }

        float yaw = Random.Range(minYaw, maxYaw);
        float pitch = Random.Range(minPitch, maxPitch);
        float force = Random.Range(minForce, maxForce);

        GameObject shuttle = Instantiate(shuttlecockPrefab, transform.position, Quaternion.identity);

        // 프리팹에 태그가 비어 있다면 강제로 부여(권장: 프리팹에서도 같은 태그 설정)
        if (!string.IsNullOrEmpty(shuttlecockTag))
            shuttle.tag = shuttlecockTag;

        var sc = shuttle.GetComponent<Shuttlecock>();
        if (sc != null)
        {
            sc.Launch(yaw, pitch, force);
        }
        else
        {
            Debug.LogWarning("[Shooter] Shuttlecock 컴포넌트가 프리팹에 없습니다.");
        }
    }
}
