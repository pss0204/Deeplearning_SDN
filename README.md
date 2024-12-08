# Optimizing SDN Packet Processing Performance Using Deep Learning

---


---

## What is SDN?
- **Definition**: Technology to control and manage networks through software.
- **Features**: Separate the **Control Plane** and **Data Plane** of the network.

### Comparison between SDN and existing networks
- **Existing Network**: Control planes are distributed to each network equipment.
- **SDN**: Provides an integrated control plane.

---

## Development Environment
- **Network Simulator**: Minnet
- **SDN Controller**: RYU
- **Deep Learning Library**: PyTorch


- <img width="869" alt="image" src="https://github.com/user-attachments/assets/a040f7a1-310d-49f5-b409-6e2bb6bffae8">


---

## Results

### Base Line
- **Conditions**: Benchmark execution based on 'simple_switch'.
- **Principle of operation**:
  - Save the source MAC address to the table.
  - If the destination MAC address is in the learning table, it is forwarded to that port.
  - Flooding packets if not learned.
- **Performance**:
  - 평균 Throughput: **3312.12 responses/s**
  - Standard deviation: **198.10 (smaller stable)**

    <img width="639" alt="image" src="https://github.com/user-attachments/assets/c819fa3d-eae5-4b72-9b8d-64688f8e49e5">


### After applying deep learning
- **Conditions**: Same test environment (128 switches, 3 hours running).
  - Using optimization techniques such as GPU porting, tiling, caching, etc.
- **Performance**:
  - 평균 Throughput: **8949.96 responses/s**
  - Standard deviation: **249.12 (smaller stable)**

  - <img width="563" alt="image" src="https://github.com/user-attachments/assets/a47ab268-9dba-48a7-8553-540dd0b44579">


    

---

## Visualization
- The resulting data can be visualized as a graph to confirm performance improvement (using e.g., Cbench).

---

## Conclusion
- Deep learning optimization of SDN significantly improves **throughput** over existing methods.
- **Proves the potential of deep learning technology in intelligent and performance optimization of network management**.
```
