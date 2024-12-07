# 기존 라이선스 및 설명...

"""
An OpenFlow 1.0 L2 learning switch implementation with batch learning using PyTorch.


Sampark.py 파일 내의 Sampark 클래스는 app_manager.RyuApp을 상속받아 
Ryu 애플리케이션으로 동작합니다. 이 애플리케이션은 PyTorch를 이용하여 

신경망 모델을 학습시키고, 이를 통해 입력 포트(in_port)와 목적지 MAC 주소(dst_mac)를 
기반으로 최적의 출력 포트를 예측합니다.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_0
from ryu.lib.mac import haddr_to_bin # 목적지 MAC 주소를 바이너리 형식으로 변환
from ryu.lib.packet import packet, ethernet, ether_types # 패킷 처리 라이브러리
from ryu.lib import hub # 멀티스레딩 지원
from packet_manager import PacketManager  # 패킷 관리 모듈 추가
import logging
from eventlet.queue import Empty  # 추가된 라인
# from Train_Sam import device

class Sampark(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_0.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(Sampark, self).__init__(*args, **kwargs)
        self.mac_to_port = {}
        self.packet_buffer = []
        self.buffer_lock = hub.Semaphore(1)
        self.prediction_queue = hub.Queue()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cache = {}  # 캐시 초기화 추가

        # 모델 초기화 및 디바이스로 이동
        self.model = self.initialize_model().to(self.device)
        try:
            self.model.load_state_dict(torch.load('switch_model.pth', map_location=self.device))
            self.logger.info("모델 로드 성공")
        except FileNotFoundError:
            self.logger.error("모델 파일을 찾을 수 없습니다.")
            exit(1)
        self.model.eval()  # 모델을 평가 모드로 설정

        # 패킷 관리자 초기화
        self.packet_manager = PacketManager('packet_data.csv')

        # 예측 작업 시작
        hub.spawn(self.prediction_worker)

        # 로그 파일 설정
        self.log_file = logging.FileHandler('sampark_training.log')
        self.log_file.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        self.logger.addHandler(self.log_file)

    def initialize_model(self):
        class SwitchModel(nn.Module):
            def __init__(self, input_size, hidden_size, num_classes):
                super(SwitchModel, self).__init__()
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(hidden_size, num_classes)

            def forward(self, x):
                out = self.fc1(x)
                out = self.relu(out)
                out = self.fc2(out)
                return out
                
        input_size = 2  # in_port와 dst_mac
        hidden_size = 64
        num_classes = 10  # 포트 수에 따라 설정
        return SwitchModel(input_size, hidden_size, num_classes)

    def encode_mac(self, mac):
        return int(mac.replace(':', ''), 16)

    def predict_out_port(self, in_port, dst_mac):
        key = (in_port, dst_mac)
        if key in self.cache:
            return self.cache[key]  # 캐시된 결과 반환

        self.model.eval()
        dst_encoded = self.encode_mac(dst_mac)
        input_tensor = torch.tensor([in_port, dst_encoded], dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(input_tensor)
            _, predicted = torch.max(output, 1)
            self.cache[key] = predicted.item()  # 예측 결과를 캐시에 저장
            return predicted.item()

    def prediction_worker(self):
        batch_size = 128  # 배치 크기 설정
        while True:
            batch = []
            for _ in range(batch_size):
                try:
                    packet = self.prediction_queue.get(timeout=1)
                    batch.append(packet)
                except hub.Timeout:
                    break
                except Empty:
                    continue 
            if batch:
                inputs = torch.tensor([[p[0], self.encode_mac(p[1])] for p in batch], dtype=torch.float32).to(self.device)
                with torch.no_grad():
                    outputs = self.model(inputs)
                    _, predicted = torch.max(outputs, 1)
                for p, port in zip(batch, predicted.tolist()):
                    in_port, dst_mac, src_mac, datapath, msg = p
                    actions = [datapath.ofproto_parser.OFPActionOutput(port)]
                    
                    data = None
                    if msg.buffer_id == datapath.ofproto.OFP_NO_BUFFER:
                        data = msg.data

                    out = datapath.ofproto_parser.OFPPacketOut(
                        datapath=datapath, buffer_id=msg.buffer_id, in_port=msg.in_port,
                        actions=actions, data=data)
                    datapath.send_msg(out)

                    # self.packet_manager.insert_packet(in_port, self.encode_mac(dst_mac), port)

                    if port != datapath.ofproto.OFPP_FLOOD:
                        self.add_flow(datapath, in_port, dst_mac, src_mac, actions)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath
        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocol(ethernet.ethernet)

        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            return

        dst = eth.dst
        src = eth.src
        in_port = msg.in_port
        dpid = datapath.id

        self.mac_to_port.setdefault(dpid, {})
        self.mac_to_port[dpid][src] = in_port

        # src를 큐에 추가
        self.prediction_queue.put((in_port, dst, src, datapath, msg))

    def add_flow(self, datapath, in_port, dst, src, actions):
        ofproto = datapath.ofproto
        match = datapath.ofproto_parser.OFPMatch(
            in_port=in_port,
            dl_dst=haddr_to_bin(dst), dl_src=haddr_to_bin(src))
        mod = datapath.ofproto_parser.OFPFlowMod(
            datapath=datapath, match=match, cookie=0,
            command=ofproto.OFPFC_ADD, idle_timeout=0, hard_timeout=0,
            priority=ofproto.OFP_DEFAULT_PRIORITY,
            flags=ofproto.OFPFF_SEND_FLOW_REM, actions=actions)
        datapath.send_msg(mod)
        