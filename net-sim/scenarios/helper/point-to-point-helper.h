#ifndef _POINT_TO_POINT_HELPER_H
#define _POINT_TO_POINT_HELPER_H

#include "ns3/point-to-point-module.h"

using namespace ns3;

// The P2PHelper acts like the ns3::PointToPointHelper,
// but sets a ns3::DropTailQueue to one packet in order to minimize queueing latency.
// Queues are simulated using a PfifoFastQueueDisc, with a default size of 100 packets.
// The queue size can be set to a custom value using SetQueueSize().
class P2PHelper : public PointToPointHelper {
public:
  P2PHelper();

  // SetQueueSize sets the queue size for the PfifoFastQueueDisc
  void SetQueueSize(StringValue);
  NetDeviceContainer Install(Ptr<Node> a, Ptr<Node> b);
private:
  StringValue queue_size_; // for the PfifoFastQueueDisc
};

#endif /* _POINT_TO_POINT_HELPER_H */
