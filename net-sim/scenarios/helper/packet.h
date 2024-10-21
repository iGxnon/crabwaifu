#ifndef _PACKET_H
#define _PACKET_H

#include "ns3/header.h"
#include "ns3/packet.h"
#include "ns3/ppp-header.h"
#include "ns3/ipv4-header.h"
#include "ns3/ipv6-header.h"
#include "ns3/udp-header.h"

using namespace ns3;
using namespace std;

bool IsUDPPacket(Ptr<Packet> p) {
    PppHeader ppp_hdr = PppHeader();
    p->RemoveHeader(ppp_hdr);
    bool is_udp = false;
    switch(ppp_hdr.GetProtocol()) {
        case 0x21: // IPv4
            {
                Ipv4Header hdr = Ipv4Header();
                p->PeekHeader(hdr);
                is_udp = hdr.GetProtocol() == 17;
            }
            break;
        case 0x57: // IPv6
            {
                Ipv6Header hdr = Ipv6Header();
                p->PeekHeader(hdr);
                is_udp = hdr.GetNextHeader() == 17;
            }
            break;
        default:
            cout << "Unknown PPP protocol: " << ppp_hdr.GetProtocol() << endl;
            break;
    }
    p->AddHeader(ppp_hdr);
    return is_udp;
}

#endif /* _PACKET_H */
