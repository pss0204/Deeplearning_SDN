#!/usr/bin/python
from mininet.net import Mininet
from mininet.node import RemoteController, OVSKernelSwitch
from mininet.topo import Topo
from mininet.cli import CLI
import time
import threading

class TreeTopo(Topo):
    def build(self, depth=1, fanout=2):
        self.hostNum = 1
        self.switchNum = 1
        self.addTree(depth, fanout)
    
    def addTree(self, depth, fanout):
        parent = self.addSwitch('s%d' % self.switchNum)
        self.switchNum += 1
        
        if depth > 0:
            for _ in range(fanout):
                child = self.addSwitch('s%d' % self.switchNum)
                self.switchNum += 1
                self.addLink(parent, child)
                self.addTree(depth-1, fanout)
        else:
            host = self.addHost('h%d' % self.hostNum)
            self.hostNum += 1
            self.addLink(host, parent)

def continuous_traffic(net):
    """1초마다 모든 호스트 페어 간 ping을 수행"""
    while True:
        hosts = net.hosts
        for h1 in hosts:
            for h2 in hosts:
                if h1 != h2:
                    # 백그라운드로 ping 실행
                    h1.cmd('ping -c 1 %s &' % h2.IP())
        time.sleep(1)  # 1초 대기

def main():
    # 트리 토폴로지 생성 (depth=3, fanout=4로 설정하면 많은 스위치 생성)
    topo = TreeTopo(depth=3, fanout=4)
    
    # 컨트롤러 및 네트워크 설정
    net = Mininet(
        topo=topo,
        controller=lambda name: RemoteController(name, ip='127.0.0.1', port=6633),
        switch=OVSKernelSwitch,
        autoSetMacs=True
    )
    
    net.start()
    
    # 트래픽 생성 스레드 시작
    traffic_thread = threading.Thread(target=continuous_traffic, args=(net,))
    traffic_thread.daemon = True  # 메인 프로그램 종료시 같이 종료
    traffic_thread.start()
    
    CLI(net)
    net.stop()

if __name__ == '__main__':
    main()