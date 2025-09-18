using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class UnderNet : MonoBehaviour
{
    public RallyJudge rallyJudge;

    private void OnTriggerEnter(Collider other)
    {
        // 셔틀콕만 감지
        if (!other.CompareTag("Shuttlecock")) return;

        // 언더넷 플래그 켜기
        rallyJudge?.MarkUnderNet();
    }
    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
