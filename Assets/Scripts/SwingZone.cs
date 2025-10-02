using System.Collections;
using System.Collections.Generic;
//using System.Diagnostics;
using UnityEngine;

public class SwingZone : MonoBehaviour
{
    public enum ZoneType { Over, Under }
    public ZoneType zoneType;

    public List<Shuttlecock> inRange = new List<Shuttlecock>();

    public List<Shuttlecock> GetShuttlecocks() => inRange;

    void OnTriggerEnter(Collider other)
    {
        Shuttlecock sc = other.GetComponentInParent<Shuttlecock>();
        if (sc == null) return;

        inRange.RemoveAll(x => x == null); // null Á¤¸®

        if (!inRange.Contains(sc))
            inRange.Add(sc);
    }

    void OnTriggerExit(Collider other)
    {
        Shuttlecock sc = other.GetComponentInParent<Shuttlecock>();
        if (sc == null) return;

        inRange.Remove(sc);
    }
}