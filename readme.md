# HiMCM 2024 B题 Q1 实操 README - U-PUE模型

## 1. Q1目标
算全球HPC年耗电，两个数都要给区间：
- `E_full`: 满载跑8760h
- `E_avg`: 实际平均情况，对上IEA 460-700TWh

## 2. 模型名
U-PUE 分类能耗核算模型
Utilization-corrected PUE Accounting Model

## 3. 公式直接用
```latex
E_full = 8760 * sum(N_j * P_peak_j * PUE_j)
E_avg = 8760 * sum(N_j * P_peak_j * PUE_j * [k_idle + (1-k_idle)*u_j])
```
j=1云/AI, 2企业/超算, 3挖矿, 4其他(网络+边缘+残差, 取总量10-15%)

变量单位：
- N_j: 台数/机架数
- P_peak_j: kW/台，机架10-40kW，AI机架80-100kW，矿机~3kW
- PUE_j: hyperscale 1.2, 平均1.55, 老旧2.0
- u_j: 云0.5-0.6, 企业0.15-0.3, AI训练0.7-0.8, 挖矿0.8-0.9
- k_idle=0.35, 8760=全年小时
- 换算: 1GW*8760h=8.76TWh

流程：`[N*P_peak] -> *8760 -> *PUE -> E_full -> *利用率修正 -> E_avg`

## 4. 三步开工
1. 查数填表：IEA Electricity 2024找460TWh基准，Uptime找PUE，Synergy找数据中心数量，NVIDIA/比特币官网找单机功率
2. 算：先Top-down倒推装机 `C_avg=460TWh/8760≈52GW`，再Bottom-up `N*P` 校验，量级对上即可
3. 出图出表：一张参数表，一张堆叠柱状图 full vs avg分4类，一张饼图看占比

## 5. 评价模型怎么写
1. Back-test: E_avg必须落在IEA区间，E_full约E_avg的1.8-2.5倍
2. 灵敏度: 每次动一个参数±10%，算 `S=(dE/E)/(dp/p)`，结论一般是 PUE>u>k_idle，画柱状图
3. 优缺点: 优点简单可扩展到Q2，缺点PUE取常数/k_idle一刀切/P_peak变化快

## 6. 输出给Q2
保存变量：`E_full, E_avg, E_j` 分4类存好
Q2直接用：`C = E_avg * sum(w_i*EF_i)`，EF: 煤0.9, 气0.4, 可再生0.03, 核0.012 kgCO2/kWh

## 7. 分工
- A: 查IEA+PUE+u填参数表
- B: 写Python计算+画图
- C: 写假设+评价+论文Q1节
