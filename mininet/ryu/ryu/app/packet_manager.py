import csv
import threading

class PacketManager:
    def __init__(self, filename, initialize=True):
        self.filename = filename
        self.lock = threading.Lock()
        if initialize:
            self.init_csv()

    def init_csv(self):
        with self.lock, open(self.filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['in_port', 'dst_mac', 'out_port'])

    def insert_packet(self, in_port, dst_mac, out_port):
        with self.lock, open(self.filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([in_port, dst_mac, out_port])

    def load_batch(self, limit):
        batch = []
        with self.lock, open(self.filename, mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # 헤더 건너뜀
            for _, row in enumerate(reader):
                if len(batch) >= limit:
                    break
                try:
                    batch.append(tuple(map(int, row)))
                except ValueError as e:
                    logging.error(f"데이터 변환 오류: {e}")
        return batch

    def delete_batch(self, limit):
        with self.lock, open(self.filename, mode='r') as file:
            lines = file.readlines()
        with self.lock, open(self.filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['in_port', 'dst_mac', 'out_port'])
            writer.writelines(lines[limit+1:])  # 헤더 제외 후 삭제