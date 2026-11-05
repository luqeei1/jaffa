| Front-end | Back-end | APE final [m] | APE online [m] | RPE 1 m [m] | End drift [%] | Map <5 cm [%] | ms / kf | Loops (correct/total) |
|---|---|---|---|---|---|---|---|---|
| Wheel odometry | Odometry only | 0.269 | 0.269 | 0.052 | 0.93 | 48 | 137 | 0/0 |
| Wheel odometry | Recommended SLAM | 0.068 | 0.139 | 0.045 | 0.08 | 67 | 558 | 207/207 |
| KISS-ICP (lidar) | Odometry only | 0.047 | 0.047 | 0.065 | 0.08 | 64 | 120 | 0/0 |
| KISS-ICP (lidar) | Recommended SLAM | 0.048 | 0.052 | 0.065 | 0.10 | 61 | 301 | 208/208 |
| RKO-LIO (lidar + IMU) | Odometry only | 0.028 | 0.028 | 0.019 | 0.06 | 53 | 121 | 0/0 |
| RKO-LIO (lidar + IMU) | Recommended SLAM | 0.027 | 0.034 | 0.018 | 0.03 | 61 | 629 | 204/204 |
| KISS-ICP + synthetic drift | Odometry only | 1.349 | 1.349 | 0.063 | 6.39 | 22 | 155 | 0/0 |
| KISS-ICP + synthetic drift | Recommended SLAM | 0.059 | 0.282 | 0.068 | 0.08 | 53 | 659 | 209/210 |
