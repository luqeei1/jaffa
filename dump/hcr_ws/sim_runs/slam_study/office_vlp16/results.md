| front-end | strategy | APE final [m] | APE online [m] | drift [%] | RPE 1 m [m] | map <5 cm [%] | loops | correct | ms / keyframe |
|---|---|---|---|---|---|---|---|---|---|
| Wheel odometry | odometry only | 0.269 | 0.269 | 0.93 | 0.052 | 48 | 0 | 0 | 137 |
| Wheel odometry | prox + LM | 0.068 | 0.139 | 0.08 | 0.045 | 67 | 207 | 207 | 428 |
| Wheel odometry | prox + iSAM2 | 0.068 | 0.139 | 0.08 | 0.045 | 67 | 207 | 207 | 413 |
| Wheel odometry | SC + LM | 3.174 | 3.790 | 0.59 | 0.082 | 26 | 113 | 90 | 404 |
| Wheel odometry | prox+SC + LM | 3.179 | 3.895 | 0.87 | 0.088 | 27 | 186 | 151 | 658 |
| Wheel odometry | prox+SC + iSAM2 + Cauchy | 0.085 | 0.138 | 0.22 | 0.051 | 64 | 225 | 207 | 538 |
| Wheel odometry | prox+SC + LM + GNC | 0.068 | 0.139 | 0.08 | 0.045 | 67 | 225 | 207 | 986 |
| Wheel odometry | SC + gate + LM | 0.085 | 0.152 | 0.11 | 0.050 | 65 | 90 | 90 | 391 |
| Wheel odometry | prox+SC + gate + iSAM2 + Cauchy | 0.083 | 0.140 | 0.18 | 0.051 | 64 | 207 | 207 | 550 |
| Wheel odometry | prox+SC + gate + LM + GNC | 0.068 | 0.139 | 0.08 | 0.045 | 67 | 207 | 207 | 558 |
| KISS-ICP | odometry only | 0.047 | 0.047 | 0.08 | 0.065 | 64 | 0 | 0 | 120 |
| KISS-ICP | prox + LM | 0.048 | 0.052 | 0.10 | 0.064 | 61 | 208 | 208 | 420 |
| KISS-ICP | prox + iSAM2 | 0.048 | 0.052 | 0.10 | 0.064 | 61 | 208 | 208 | 412 |
| KISS-ICP | SC + LM | 3.409 | 4.117 | 4.25 | 0.715 | 20 | 113 | 92 | 454 |
| KISS-ICP | prox+SC + LM | 3.342 | 4.482 | 4.92 | 0.755 | 22 | 199 | 163 | 551 |
| KISS-ICP | prox+SC + iSAM2 + Cauchy | 0.047 | 0.050 | 0.09 | 0.063 | 64 | 225 | 208 | 473 |
| KISS-ICP | prox+SC + LM + GNC | 0.048 | 0.052 | 0.10 | 0.065 | 61 | 225 | 208 | 658 |
| KISS-ICP | SC + gate + LM | 0.050 | 0.054 | 0.06 | 0.065 | 58 | 92 | 92 | 246 |
| KISS-ICP | prox+SC + gate + iSAM2 + Cauchy | 0.047 | 0.050 | 0.09 | 0.063 | 64 | 208 | 208 | 321 |
| KISS-ICP | prox+SC + gate + LM + GNC | 0.048 | 0.052 | 0.10 | 0.065 | 61 | 208 | 208 | 301 |
| RKO-LIO | odometry only | 0.028 | 0.028 | 0.06 | 0.019 | 53 | 0 | 0 | 121 |
| RKO-LIO | prox + LM | 0.027 | 0.034 | 0.03 | 0.018 | 61 | 204 | 204 | 518 |
| RKO-LIO | prox + iSAM2 | 0.027 | 0.033 | 0.03 | 0.018 | 61 | 204 | 204 | 548 |
| RKO-LIO | SC + LM | 3.476 | 4.237 | 3.49 | 1.175 | 19 | 109 | 89 | 539 |
| RKO-LIO | prox+SC + LM | 3.356 | 4.262 | 4.06 | 1.226 | 20 | 194 | 156 | 712 |
| RKO-LIO | prox+SC + iSAM2 + Cauchy | 0.028 | 0.030 | 0.05 | 0.018 | 57 | 221 | 204 | 618 |
| RKO-LIO | prox+SC + LM + GNC | 0.027 | 0.034 | 0.03 | 0.018 | 61 | 221 | 204 | 998 |
| RKO-LIO | SC + gate + LM | 0.028 | 0.033 | 0.03 | 0.019 | 60 | 89 | 89 | 501 |
| RKO-LIO | prox+SC + gate + iSAM2 + Cauchy | 0.028 | 0.030 | 0.05 | 0.018 | 57 | 204 | 204 | 615 |
| RKO-LIO | prox+SC + gate + LM + GNC | 0.027 | 0.034 | 0.03 | 0.018 | 61 | 204 | 204 | 629 |
| KISS-ICP + drift (synthetic) | odometry only | 1.349 | 1.349 | 6.39 | 0.063 | 22 | 0 | 0 | 155 |
| KISS-ICP + drift (synthetic) | prox + LM | 0.061 | 0.266 | 0.08 | 0.066 | 52 | 210 | 209 | 558 |
| KISS-ICP + drift (synthetic) | prox + iSAM2 | 0.060 | 0.266 | 0.07 | 0.066 | 53 | 210 | 209 | 545 |
| KISS-ICP + drift (synthetic) | SC + LM | 3.457 | 4.109 | 3.33 | 0.722 | 21 | 115 | 93 | 556 |
| KISS-ICP + drift (synthetic) | prox+SC + LM | 3.317 | 4.406 | 3.75 | 0.762 | 23 | 198 | 159 | 890 |
| KISS-ICP + drift (synthetic) | prox+SC + iSAM2 + Cauchy | 0.104 | 0.586 | 0.44 | 0.063 | 38 | 220 | 196 | 772 |
| KISS-ICP + drift (synthetic) | prox+SC + LM + GNC | 0.059 | 0.286 | 0.08 | 0.068 | 52 | 228 | 209 | 1161 |
| KISS-ICP + drift (synthetic) | SC + gate + LM | 0.063 | 0.280 | 0.03 | 0.064 | 47 | 93 | 93 | 524 |
| KISS-ICP + drift (synthetic) | prox+SC + gate + iSAM2 + Cauchy | 0.100 | 0.602 | 0.41 | 0.063 | 38 | 199 | 198 | 654 |
| KISS-ICP + drift (synthetic) | prox+SC + gate + LM + GNC | 0.059 | 0.282 | 0.08 | 0.068 | 53 | 210 | 209 | 659 |
