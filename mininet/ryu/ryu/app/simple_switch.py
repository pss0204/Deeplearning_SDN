# Copyright (C) 2011 Nippon Telegraph and Telephone Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
An OpenFlow 1.0 L2 learning switch implementation.
"""


from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_0
from ryu.lib.mac import haddr_to_bin
from ryu.lib.packet import packet
from ryu.lib.packet import ethernet
from ryu.lib.packet import ether_types
from packet_manager import PacketManager


class SimpleSwitch(app_manager.RyuApp): # 클래스 상속 
    OFP_VERSIONS = [ofproto_v1_0.OFP_VERSION] # OpenFlow 1.0

    def __init__(self, *args, **kwargs): # 초기화 매서드 클래스 생성시에 자동으로 호출됨
        super(SimpleSwitch, self).__init__(*args, **kwargs) #부모클래스 초기화 
        self.mac_to_port = {} # mac 주소와 포트 매핑 테이블 ->딕셔너리임 # {'dpid1': {'mac1': port1, 'mac2': port2}, 'dpid2': {...}, ...}
        self.packet_manager = PacketManager('packet_data.csv')
    def add_flow(self, datapath, in_port, dst, src, actions): #플로우 추가 함수  
    #특정 패킷 흐름에 대한 매치 룰과 수행할 액션을 정의하여 스위치에 플로우를 설치합니다.
    #예를 들어, 특정 출발지와 목적지 MAC 주소, 입력 포트를 기준으로 패킷을 특정 포트로 출력하는 규칙을 추가합니다.

    #datapath : 스위치 데이터 패스 객체 
    #in_port : 패킷이 들어온 포트 번호 
    #dst : 목적지 mac 주소 
    #src : 출발지 mac 주소 
    #actions : 패킷을 처리하기 위한 액션 목록 

        ofproto = datapath.ofproto

        match = datapath.ofproto_parser.OFPMatch(
            in_port=in_port,
            dl_dst=haddr_to_bin(dst), dl_src=haddr_to_bin(src))
        # 플로우 모드 메시지 생성 
        mod = datapath.ofproto_parser.OFPFlowMod(
            datapath=datapath, match=match, cookie=0,
            command=ofproto.OFPFC_ADD, idle_timeout=0, hard_timeout=0,
            priority=ofproto.OFP_DEFAULT_PRIORITY,
            flags=ofproto.OFPFF_SEND_FLOW_REM, actions=actions)
        datapath.send_msg(mod)

    def encode_mac(self, mac): # 맥주소 인코딩 
        return int(mac.replace(':', ''), 16)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):

        
    # 스위치에서 패킷이 도착하면 Packet-In 이벤트가 발생하여 이 핸들러가 호출됩니다.
    # 패킷의 출발지와 목적지 MAC 주소를 추출하고, 출발지 MAC 주소를 학습 테이블에 저장합니다.
    # 목적지 MAC 주소가 학습 테이블에 있으면 해당 포트로 패킷을 전달하고, 없으면 플러딩합니다.
    # 필요한 경우 플로우 룰을 설치하여 동일한 패킷에 대해 이후에는 패킷 인 이벤트가 발생하지 않도록 설정합니다.
    #        
    
        # 1. 기본 정보 추출 
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto

        # 2. 패킷 데이터 추출 
        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocol(ethernet.ethernet)

        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            # ignore lldp packet
            return
        dst = eth.dst # 목적지 mac 주소 
        src = eth.src # 출발지 mac 주소 

        dpid = datapath.id
        #3. mac 주소와 포트 매핑 테이블 생성 (학습)
        self.mac_to_port.setdefault(dpid, {}) 

        self.logger.info("packet in %s %s %s %s", dpid, src, dst, msg.in_port)

        # learn a mac address to avoid FLOOD next time.
        self.mac_to_port[dpid][src] = msg.in_port

        # 4. 포워딩 결정    
        if dst in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][dst]
        else:
            out_port = ofproto.OFPP_FLOOD # 목적지 mac 주소가 없으면 플로딩 

        actions = [datapath.ofproto_parser.OFPActionOutput(out_port)]

        # 5. 플로우 룰 설치 
        if out_port != ofproto.OFPP_FLOOD:
            self.add_flow(datapath, msg.in_port, dst, src, actions)

            
            

        data = None
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data

        out = datapath.ofproto_parser.OFPPacketOut(
            datapath=datapath, buffer_id=msg.buffer_id, in_port=msg.in_port,
            actions=actions, data=data)
        datapath.send_msg(out)

        # 패킷 정보를 데이터베이스에 저장
        self.packet_manager.insert_packet(msg.in_port, self.encode_mac(dst), out_port)


    @set_ev_cls(ofp_event.EventOFPPortStatus, MAIN_DISPATCHER)
    def _port_status_handler(self, ev):
        # 1. 포트 상태 이벤트 처리 
        msg = ev.msg
        reason = msg.reason
        port_no = msg.desc.port_no

        ofproto = msg.datapath.ofproto
        if reason == ofproto.OFPPR_ADD:
            self.logger.info("port added %s", port_no)
        elif reason == ofproto.OFPPR_DELETE:
            self.logger.info("port deleted %s", port_no)
        elif reason == ofproto.OFPPR_MODIFY:
            self.logger.info("port modified %s", port_no)
        else:
            self.logger.info("Illeagal port state %s %s", port_no, reason)
