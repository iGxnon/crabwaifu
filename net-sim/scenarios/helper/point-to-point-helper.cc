#include "ns3/core-module.h"
#include "ns3/traffic-control-helper.h"
#include "ns3/ipv4-address-helper.h"
#include "ns3/ipv6-address-helper.h"
#include "point-to-point-helper.h"

using namespace ns3;

P2PHelper::P2PHelper() : queue_size_(StringValue("100p")) {
  SetQueue("ns3::DropTailQueue", "MaxSize", StringValue("1p"));
}

void P2PHelper::SetQueueSize(StringValue size) {
  queue_size_ = size;
}

NetDeviceContainer P2PHelper::Install(Ptr<Node> a, Ptr<Node> b) {
  NetDeviceContainer devices = PointToPointHelper::Install(a, b);
  TrafficControlHelper tch;
  tch.SetRootQueueDisc("ns3::PfifoFastQueueDisc", "MaxSize", queue_size_);
  tch.Install(devices);

  Ipv4AddressHelper ipv4;
  ipv4.SetBase("193.167.50.0", "255.255.255.0");
  ipv4.Assign(devices);

  Ipv6AddressHelper ipv6;
  ipv6.SetBase("fd00:cafe:cafe:50::", 64);
  ipv6.Assign(devices);

  return devices;
}
