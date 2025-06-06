using System.Collections.Generic; // List ��뿡 �ʿ�
using UnityEngine;


public class RealTimeController : MonoBehaviour
{
    [SerializeField] private SkeletonMapper.Basis skeletonMapper;   // ��ü ������ ���� ������ Basis?
    [SerializeField] private List<Listener.Basis> listener;         // ���� ������ �޴� ����? Ȯ���ʿ�

    // Update is called once per frame
    private void Update()
    {
        for (var i = 0; i < listener.Count; i++) listener[i].MoveBoneMap(skeletonMapper.GetBoneMap());
    }
}