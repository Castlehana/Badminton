using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public enum RallyState
{
    Ready, Rallying, Ended
}

public class RallyManager : MonoBehaviour
{
    public RallyState State;

    private bool isResetting = false;

    // Start is called before the first frame update
    void Start()
    {
        State = RallyState.Ready;
    }

    // Update is called once per frame
    void Update()
    {
        if (State == RallyState.Ended && !isResetting)
        {
            StartCoroutine(ReturnToReady());
        }
    }

    private IEnumerator ReturnToReady()
    {
        isResetting = true;

        yield return new WaitForSeconds(1.0f);

        // test here
        //State = RallyState.Ready;

        isResetting = false;
    }
}
