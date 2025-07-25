using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using UnityEngine;

namespace Listener
{
    public class DualApp : Basis
    {
        private static readonly int
            PositionsId = Shader.PropertyToID("_Positions"),
            ScaleId = Shader.PropertyToID("_Scale"),
            UarmId = Shader.PropertyToID("_Uarm"),
            OrigId = Shader.PropertyToID("_Origin"),
            LarmId = Shader.PropertyToID("_Larm"),
            HandId = Shader.PropertyToID("_Hand");

        [SerializeField] private Material mcPredMaterial;
        [SerializeField] private Mesh mcPredMesh;
        [SerializeField] private GameObject upperArm;
        [SerializeField] private int port = 50003;
        [SerializeField] private bool leftHandMode = true;

        private byte[] _msgTail;
        private ComputeBuffer _positionsBuffer;
        private Bounds _bounds = new(Vector3.zero, Vector3.one * 5f);
        private int _cubeCount;

        private Thread _udpListenerThread;
        private UdpClient _udpClient;
        private bool _pause;

        private Quaternion _handRot;
        private Vector3 _handPos;
        private Quaternion _larmRot;
        private Vector3 _larmPos;
        private Quaternion _uarmRot;


        // Start is called before the first frame update
        private void Start()
        {
            _udpListenerThread = new Thread(UDPListener);
            _udpListenerThread.Start();
            mcPredMaterial.SetFloat(ScaleId, 0.01f);
        }

        private void UDPListener()
        {
            //Creates a UdpClient for reading incoming data.
            _udpClient = new UdpClient(port);
            //Creates an IPEndPoint to record the IP Address and port number of the sender.
            var remoteIpEndPoint = new IPEndPoint(IPAddress.Any, 0);

            Debug.Log("start listening");
            while (true)
            {
                if (_pause)
                {
                    Thread.Sleep(100);
                    continue;
                }

                // Blocks until a message returns on this socket from a remote host.
                var msg = _udpClient.Receive(ref remoteIpEndPoint);

                // the basic message
                _handRot = new Quaternion(
                    BitConverter.ToSingle(msg, 4),
                    BitConverter.ToSingle(msg, 8),
                    BitConverter.ToSingle(msg, 12),
                    BitConverter.ToSingle(msg, 0)
                );
                _handPos = new Vector3(
                    BitConverter.ToSingle(msg, 16),
                    BitConverter.ToSingle(msg, 20),
                    BitConverter.ToSingle(msg, 24)
                );
                _larmRot = new Quaternion(
                    BitConverter.ToSingle(msg, 32),
                    BitConverter.ToSingle(msg, 36),
                    BitConverter.ToSingle(msg, 40),
                    BitConverter.ToSingle(msg, 28)
                );
                _larmPos = new Vector3(
                    BitConverter.ToSingle(msg, 44),
                    BitConverter.ToSingle(msg, 48),
                    BitConverter.ToSingle(msg, 52)
                );
                _uarmRot = new Quaternion(
                    BitConverter.ToSingle(msg, 60),
                    BitConverter.ToSingle(msg, 64),
                    BitConverter.ToSingle(msg, 68),
                    BitConverter.ToSingle(msg, 56)
                );

                // if the message is longer, we have additional monte carlo predictions
                // store tail to pass to compute shader
                if (msg.Length > 72)
                    _msgTail = msg.Skip(72).ToArray();
            }
        }

        public override void MoveBoneMap(Dictionary<string, GameObject> boneMap)
        {
            var uaPos = upperArm.transform.position;

            if (leftHandMode)
            {
                boneMap["LeftHand"].transform.SetPositionAndRotation(_handPos + uaPos, _handRot);
                boneMap["LeftLowerArm"].transform.SetPositionAndRotation(_larmPos + uaPos, _larmRot);
                boneMap["LeftUpperArm"].transform.SetPositionAndRotation(uaPos, _uarmRot);
            }
            else
            {
                boneMap["RightHand"].transform.SetPositionAndRotation(_handPos + uaPos, _handRot);
                boneMap["RightLowerArm"].transform.SetPositionAndRotation(_larmPos + uaPos, _larmRot);
                boneMap["RightUpperArm"].transform.SetPositionAndRotation(uaPos, _uarmRot);
            }


            // now the positions buffer for our monte carlo hand and larm positions
            if (_msgTail is not null)
            {
                // create a new buffer if non was created yet or if the size changed
                if (_positionsBuffer is null || _positionsBuffer.count != _msgTail.Count())
                {
                    _positionsBuffer = new ComputeBuffer(_msgTail.Count(), 3 * 4);
                    _cubeCount = _positionsBuffer.count / 12;
                    Debug.Log("Created positions buffer with count " +
                              _positionsBuffer.count + " (" + _cubeCount +
                              " cubes)");
                }

                // write positions to buffer
                _positionsBuffer.SetData(_msgTail);

                // we add the shoulder position to predicted positions on the GPU 
                mcPredMaterial.SetVector(OrigId, uaPos);
                mcPredMaterial.SetVector(UarmId, uaPos);
                mcPredMaterial.SetVector(LarmId, _larmPos);
                mcPredMaterial.SetVector(HandId, _handPos);
                mcPredMaterial.SetBuffer(PositionsId, _positionsBuffer);

                // instantiate cubes 
                Graphics.DrawMeshInstancedProcedural(
                    mcPredMesh, 0, mcPredMaterial, _bounds, _cubeCount
                );
                // for finding reason of motion
                Debug.Log($"[UDP 수신 데이터] UarmRot: {_uarmRot}, LarmRot: {_larmRot}, HandRot: {_handRot}");
                Debug.Log($"[UDP 수신 위치] UarmPos: {_larmPos}, HandPos: {_handPos}");
            }
        }

        private void Update()
        {
            if (Input.GetKeyDown("space"))
            {
                _pause = !_pause;
                Debug.Log("Pause " + _pause);
            }
        }
    }
}