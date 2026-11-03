# D-GLOBAL-WHERE P2 Part 5: criteria-aware WHERE, N = 1,000,000

Every DuckDB count and EXISTS answer equals the scalar oracle in cases.json (checked before this file was written).

Population targets NOT met by the generator (achieved is what is measured; never assumed from K): k1_tiny_single (target 50, achieved 1,000), k3_tiny_and (target 50, achieved 765), k3_tiny_or (target 50, achieved 999), k3_tiny_mixed (target 50, achieved 428), k8_tiny_or (target 50, achieved 999), k17_tiny_or (target 50, achieved 1,000), k24_tiny_or (target 50, achieved 1,001)

DuckDB build + ANALYZE: threads 1: 433,303 us (build 432,942, analyze 362); threads 4: 408,392 us (build 407,967, analyze 425)

### N = 1,000,000, DuckDB threads = 1

| K group | case | pop | shape | achieved | lower plain/fused us | ours count us (order) | ours any us | ours rows us | duck count wall us | duck count engine us | duck exists wall us | count wall ratio | count wall | count engine ratio | count engine | any ratio | any |
|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|---:|---|
| 1 | k1_tiny_single | tiny | single | 1,000 | 0.10/0.11 | 140.6 (skip_fused) | 142.7 | 146.1 | 1918.3 | 1981.6 | 589.4 | 13.64 | WIN | 14.09 | WIN | 4.13 | WIN |
| 1 | k1_medium_single | medium | single | 10,000 | 0.10/0.11 | 138.6 (skip) | 137.9 | 145.3 | 1739.0 | 1799.0 | 852.7 | 12.55 | WIN | 12.98 | WIN | 6.18 | WIN |
| 1 | k1_large_single | large | single | 500,005 | 0.10/0.11 | 140.5 (plain) | 145.2 | 145.9 | 2580.9 | 2118.7 | 744.5 | 18.37 | WIN | 15.08 | WIN | 5.13 | WIN |
| 2-4 | k3_tiny_and | tiny | and | 765 | 0.28/0.51 | 567.8 (skip_fused) | 567.7 | 597.0 | 4638.3 | 4933.6 | 842.4 | 8.17 | WIN | 8.69 | WIN | 1.48 | WIN |
| 2-4 | k3_medium_and | medium | and | 10,139 | 0.34/0.69 | 528.7 (skip_fused) | 625.5 | 610.0 | 3490.4 | 4234.6 | 902.7 | 6.60 | WIN | 8.01 | WIN | 1.44 | WIN |
| 2-4 | k3_large_and | large | and | 499,628 | 0.40/0.85 | 531.4 (fused) | 524.5 | 563.7 | 3341.9 | 3411.8 | 874.7 | 6.29 | WIN | 6.42 | WIN | 1.67 | WIN |
| 2-4 | k3_tiny_or | tiny | or | 999 | 0.28/0.57 | 582.3 (plain) | 577.6 | 585.8 | 2403.1 | 2950.3 | 965.6 | 4.13 | WIN | 5.07 | WIN | 1.67 | WIN |
| 2-4 | k3_medium_or | medium | or | 10,000 | 0.28/0.56 | 557.8 (skip_fused) | 547.4 | 560.1 | 6476.5 | 6880.9 | 691.1 | 11.61 | WIN | 12.34 | WIN | 1.26 | WIN |
| 2-4 | k3_large_or | large | or | 500,615 | 0.28/0.57 | 529.7 (skip) | 520.3 | 531.2 | 21854.9 | 20115.1 | 728.7 | 41.26 | WIN | 37.97 | WIN | 1.40 | WIN |
| 2-4 | k3_tiny_mixed | tiny | mixed | 428 | 0.37/0.65 | 547.4 (skip_fused) | 546.1 | 552.2 | 3714.8 | 3736.2 | 730.0 | 6.79 | WIN | 6.83 | WIN | 1.34 | WIN |
| 2-4 | k3_medium_mixed | medium | mixed | 9,819 | 0.37/0.65 | 536.8 (skip_fused) | 529.3 | 542.7 | 4009.3 | 4073.9 | 721.6 | 7.47 | WIN | 7.59 | WIN | 1.36 | WIN |
| 2-4 | k3_large_mixed | large | mixed | 499,424 | 0.36/0.65 | 516.1 (skip_fused) | 509.3 | 523.2 | 15868.0 | 14564.6 | 1095.7 | 30.74 | WIN | 28.22 | WIN | 2.15 | WIN |
| 5-16 | k8_tiny_and | tiny | and | 46 | 0.53/2.34 | 959.4 (plain) | 965.7 | 964.9 | 4496.9 | 4455.4 | 1017.4 | 4.69 | WIN | 4.64 | WIN | 1.05 | PARITY |
| 5-16 | k8_medium_and | medium | and | 10,097 | 0.52/2.38 | 1362.3 (fused) | 1365.2 | 1378.7 | 6335.4 | 6355.1 | 881.7 | 4.65 | WIN | 4.66 | WIN | 0.65 | LOSS |
| 5-16 | k8_large_and | large | and | 498,124 | 0.63/2.43 | 1433.0 (fused) | 1425.0 | 1498.7 | 7755.4 | 8069.1 | 881.5 | 5.41 | WIN | 5.63 | WIN | 0.62 | LOSS |
| 5-16 | k8_tiny_or | tiny | or | 999 | 0.51/2.63 | 1430.3 (skip_fused) | 1422.4 | 1433.4 | 3547.2 | 3593.4 | 870.5 | 2.48 | WIN | 2.51 | WIN | 0.61 | LOSS |
| 5-16 | k8_medium_or | medium | or | 9,997 | 0.76/4.19 | 1530.3 (fused) | 1484.5 | 1519.6 | 16571.5 | 17463.0 | 954.4 | 10.83 | WIN | 11.41 | WIN | 0.64 | LOSS |
| 5-16 | k8_large_or | large | or | 498,831 | 0.51/2.61 | 1417.1 (fused) | 1408.1 | 1458.2 | 30636.8 | 30176.8 | 946.6 | 21.62 | WIN | 21.29 | WIN | 0.67 | LOSS |
| 5-16 | k8_tiny_mixed | tiny | mixed | 91 | 0.87/2.65 | 1429.4 (skip) | 1385.6 | 1394.0 | 14878.0 | 14745.0 | 1692.6 | 10.41 | WIN | 10.32 | WIN | 1.22 | WIN |
| 5-16 | k8_medium_mixed | medium | mixed | 9,901 | 0.86/2.61 | 1399.0 (skip_fused) | 1390.1 | 1440.9 | 27575.4 | 28051.8 | 1012.8 | 19.71 | WIN | 20.05 | WIN | 0.73 | LOSS |
| 5-16 | k8_large_mixed | large | mixed | 500,157 | 0.85/2.62 | 1492.9 (skip_fused) | 1528.4 | 1475.3 | 74454.8 | 75137.7 | 1078.1 | 49.87 | WIN | 50.33 | WIN | 0.71 | LOSS |
| 17+ | k17_tiny_and | tiny | and | 45 | 1.07/7.44 | 2839.3 (skip_fused) | 2888.1 | 2961.9 | 15133.9 | 20282.9 | 2743.6 | 5.33 | WIN | 7.14 | WIN | 0.95 | PARITY |
| 17+ | k17_medium_and | medium | and | 9,754 | 1.06/7.98 | 2909.6 (skip_fused) | 2879.3 | 2873.7 | 17023.7 | 15953.3 | 1162.3 | 5.85 | WIN | 5.48 | WIN | 0.40 | LOSS |
| 17+ | k17_large_and | large | and | 499,819 | 1.06/7.44 | 2776.0 (skip_fused) | 2764.1 | 2786.9 | 20204.8 | 20222.8 | 1175.4 | 7.28 | WIN | 7.28 | WIN | 0.43 | LOSS |
| 17+ | k17_tiny_or | tiny | or | 1,000 | 1.07/7.98 | 2835.1 (skip_fused) | 2854.0 | 2956.5 | 4772.1 | 5143.0 | 1062.8 | 1.68 | WIN | 1.81 | WIN | 0.37 | LOSS |
| 17+ | k17_medium_or | medium | or | 10,000 | 1.04/8.23 | 2853.3 (plain) | 2769.2 | 2996.4 | 22157.3 | 22382.8 | 1163.2 | 7.77 | WIN | 7.84 | WIN | 0.42 | LOSS |
| 17+ | k17_large_or | large | or | 505,114 | 1.06/8.19 | 2798.5 (skip) | 2772.4 | 2785.1 | 46298.8 | 55727.1 | 1836.9 | 16.54 | WIN | 19.91 | WIN | 0.66 | LOSS |
| 17+ | k17_tiny_mixed | tiny | mixed | 46 | 1.91/6.99 | 2378.1 (skip) | 2308.0 | 1620.0 | 14384.5 | 14225.7 | 2724.9 | 6.05 | WIN | 5.98 | WIN | 1.18 | WIN |
| 17+ | k17_medium_mixed | medium | mixed | 10,098 | 2.92/10.53 | 3252.1 (fused) | 3559.2 | 3580.7 | 32478.6 | 31778.7 | 1308.0 | 9.99 | WIN | 9.77 | WIN | 0.37 | LOSS |
| 17+ | k17_large_mixed | large | mixed | 498,269 | 1.91/6.99 | 2838.3 (fused) | 2916.9 | 2881.7 | 139929.3 | 137801.1 | 1552.6 | 49.30 | WIN | 48.55 | WIN | 0.53 | LOSS |
| 17+ | k24_tiny_and | tiny | and | 50 | 1.48/15.64 | 3822.1 (skip_fused) | 3885.5 | 3864.8 | 14418.2 | 14223.2 | 1653.5 | 3.77 | WIN | 3.72 | WIN | 0.43 | LOSS |
| 17+ | k24_medium_and | medium | and | 9,900 | 1.44/15.57 | 3980.4 (skip_fused) | 3775.8 | 3769.6 | 21674.5 | 26902.6 | 1422.5 | 5.45 | WIN | 6.76 | WIN | 0.38 | LOSS |
| 17+ | k24_large_and | large | and | 502,071 | 1.42/15.70 | 4162.7 (skip_fused) | 3867.2 | 3806.4 | 25627.8 | 25957.0 | 1403.9 | 6.16 | WIN | 6.24 | WIN | 0.36 | LOSS |
| 17+ | k24_tiny_or | tiny | or | 1,001 | 1.46/16.35 | 3725.8 (skip) | 3815.8 | 3741.7 | 5061.8 | 5118.6 | 1249.9 | 1.36 | WIN | 1.37 | WIN | 0.33 | LOSS |
| 17+ | k24_medium_or | medium | or | 9,995 | 2.18/24.74 | 3713.5 (skip_fused) | 3723.3 | 3736.2 | 23987.1 | 24625.8 | 1328.4 | 6.46 | WIN | 6.63 | WIN | 0.36 | LOSS |
| 17+ | k24_large_or | large | or | 495,651 | 1.44/16.39 | 3749.3 (plain) | 3784.1 | 3719.0 | 54222.9 | 81416.4 | 2195.6 | 14.46 | WIN | 21.72 | WIN | 0.58 | LOSS |
| 17+ | k24_tiny_mixed | tiny | mixed | 48 | 2.62/11.04 | 3872.5 (skip_fused) | 3859.8 | 3999.1 | 77685.0 | 74593.2 | 1816.2 | 20.06 | WIN | 19.26 | WIN | 0.47 | LOSS |
| 17+ | k24_medium_mixed | medium | mixed | 9,953 | 2.56/10.94 | 3732.2 (skip_fused) | 3833.5 | 4130.1 | 105626.4 | 102196.2 | 1680.8 | 28.30 | WIN | 27.38 | WIN | 0.44 | LOSS |
| 17+ | k24_large_mixed | large | mixed | 500,471 | 2.56/10.79 | 3885.2 (fused) | 3820.9 | 3818.3 | 210186.7 | 206403.2 | 2679.3 | 54.10 | WIN | 53.13 | WIN | 0.70 | LOSS |
| 2-4 | adv_one_everything | medium | and | 10,000 | 0.21/0.37 | 284.8 (skip) | 295.7 | 301.6 | 1736.4 | 1791.5 | 629.8 | 6.10 | WIN | 6.29 | WIN | 2.13 | WIN |
| 2-4 | adv_one_nothing | medium | and | 1,002 | 0.29/0.50 | 290.0 (plain) | 285.5 | 288.5 | 3253.2 | 3392.3 | 913.9 | 11.22 | WIN | 11.70 | WIN | 3.20 | WIN |
| 2-4 | adv_or_one_everything | large | or | 1,000,000 | 0.21/0.41 | 333.9 (skip_fused) | 314.0 | 328.3 | 568.5 | - | 635.3 | 1.70 | WIN | - | n/a | 2.02 | WIN |
| 2-4 | adv_or_one_nothing | medium | or | 1,000 | 0.21/0.40 | 316.1 (skip_fused) | 320.7 | 327.7 | 2203.6 | 2167.9 | 637.7 | 6.97 | WIN | 6.86 | WIN | 1.99 | WIN |
| 5-16 | adv_16_nonselective | large | and | 827,152 | 0.96/6.75 | 2742.0 (fused) | 2762.9 | 2774.1 | 21284.3 | 29563.1 | 1162.1 | 7.76 | WIN | 10.78 | WIN | 0.42 | LOSS |
| 5-16 | adv_16_collapse_one | tiny | and | 46 | 0.99/6.59 | 958.5 (skip) | 966.3 | 985.8 | 6147.6 | 6055.4 | 1327.0 | 6.41 | WIN | 6.32 | WIN | 1.37 | WIN |
| 2-4 | adv_correlated_2 | large | and | 500,001 | 0.22/0.38 | 316.3 (fused) | 312.5 | 321.9 | 2401.5 | 2445.3 | 598.1 | 7.59 | WIN | 7.73 | WIN | 1.91 | WIN |
| 5-16 | adv_correlated_5 | large | and | 490,001 | 0.39/1.16 | 612.3 (skip_fused) | 633.6 | 618.3 | 2414.2 | 2579.0 | 652.2 | 3.94 | WIN | 4.21 | WIN | 1.03 | PARITY |
| 2-4 | adv_contradictory_2 | empty | and | 0 | 0.31/0.61 | 271.8 (fused) | 267.4 | 266.5 | 349.8 | 492.8 | 464.3 | 1.29 | WIN | 1.81 | WIN | 1.74 | WIN |
| 17+ | adv_contradictory_17 | empty | and | 0 | 1.04/7.62 | 657.5 (skip) | 665.4 | 697.2 | 602.0 | 701.8 | 740.1 | 0.92 | PARITY | 1.07 | PARITY | 1.11 | WIN |
| 2-4 | null_and | large | and | 99,875 | 0.32/0.50 | 346.4 (skip_fused) | 339.5 | 347.6 | 2862.6 | 3037.3 | 679.7 | 8.26 | WIN | 8.77 | WIN | 2.00 | WIN |
| 1 | null_not | large | not | 450,007 | 0.24/0.45 | 186.5 (fused) | 186.3 | 185.5 | 2190.0 | 2244.8 | 570.5 | 11.74 | WIN | 12.03 | WIN | 3.06 | WIN |
| 2-4 | null_or | large | or | 190,006 | 0.36/0.53 | 345.3 (plain) | 356.0 | 352.2 | 9313.9 | 9174.3 | 662.5 | 26.97 | WIN | 26.57 | WIN | 1.86 | WIN |
| 2-4 | null_not_and | large | not | 850,017 | 0.45/0.85 | 366.1 (skip_fused) | 349.9 | 365.7 | 26557.3 | 27339.2 | 731.3 | 72.53 | WIN | 74.67 | WIN | 2.09 | WIN |
| 5-16 | null_and_5 | large | and | 267,454 | 0.61/1.50 | 931.4 (skip_fused) | 937.4 | 949.4 | 4991.5 | 5394.2 | 802.3 | 5.36 | WIN | 5.79 | WIN | 0.86 | LOSS |

Representation: Count and Any are `virtual` (no mask survives the call); Rows is `transient` (one Keep mask of 125,000 bytes held for the call); `persistent` never occurs here. Ours rows has no DuckDB counterpart.

Summary, N = 1,000,000, threads = 1 (regular grid only: 39 cases; 15 adversarial/nullable cases excluded from the figures below)

| comparison | weighted ours us | weighted duck us | weighted ratio | weighted | unweighted ours us | unweighted duck us | unweighted ratio | unweighted | geo-mean ratio |
|---|---:|---:|---:|---|---:|---:|---:|---|---:|
| count vs duck count, wall | 692.0 | 9803.4 | 14.17 | WIN | 1997.3 | 28440.0 | 14.24 | WIN | 9.55 |
| count vs duck count, engine | 692.0 | 9860.8 | 14.25 | WIN | 1997.3 | 29328.2 | 14.68 | WIN | 9.89 |
| any vs duck exists, wall | 693.9 | 879.6 | 1.27 | WIN | 1995.5 | 1262.4 | 0.63 | LOSS | 0.83 |

Per K group (count vs duck count, wall; mean us over regular cases):

| K group | cases | ours us | duck us | ratio | verdict | weight |
|---|---:|---:|---:|---:|---|---:|
| 1 | 3 | 139.9 | 2079.4 | 14.86 | WIN | 0.34 |
| 2-4 | 9 | 544.2 | 7310.8 | 13.43 | WIN | 0.46 |
| 5-16 | 9 | 1383.8 | 20694.6 | 14.96 | WIN | 0.14 |
| 17+ | 18 | 3340.2 | 47270.7 | 14.15 | WIN | 0.06 |

Ours count >= 1000 us (FAILED-GATE on the substrate path, reported): 27 of 54 cases.

### N = 1,000,000, DuckDB threads = 4

| K group | case | pop | shape | achieved | lower plain/fused us | ours count us (order) | ours any us | ours rows us | duck count wall us | duck count engine us | duck exists wall us | count wall ratio | count wall | count engine ratio | count engine | any ratio | any |
|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|---:|---|
| 1 | k1_tiny_single | tiny | single | 1,000 | 0.10/0.11 | 140.6 (skip_fused) | 142.7 | 146.1 | 1139.3 | 1264.4 | 927.9 | 8.10 | WIN | 8.99 | WIN | 6.50 | WIN |
| 1 | k1_medium_single | medium | single | 10,000 | 0.10/0.11 | 138.6 (skip) | 137.9 | 145.3 | 1194.6 | 1195.2 | 875.1 | 8.62 | WIN | 8.63 | WIN | 6.35 | WIN |
| 1 | k1_large_single | large | single | 500,005 | 0.10/0.11 | 140.5 (plain) | 145.2 | 145.9 | 1018.2 | 1065.5 | 967.3 | 7.25 | WIN | 7.58 | WIN | 6.66 | WIN |
| 2-4 | k3_tiny_and | tiny | and | 765 | 0.28/0.51 | 567.8 (skip_fused) | 567.7 | 597.0 | 1775.7 | 1627.2 | 1010.9 | 3.13 | WIN | 2.87 | WIN | 1.78 | WIN |
| 2-4 | k3_medium_and | medium | and | 10,139 | 0.34/0.69 | 528.7 (skip_fused) | 625.5 | 610.0 | 1711.6 | 1854.9 | 1328.7 | 3.24 | WIN | 3.51 | WIN | 2.12 | WIN |
| 2-4 | k3_large_and | large | and | 499,628 | 0.40/0.85 | 531.4 (fused) | 524.5 | 563.7 | 1989.3 | 2067.8 | 1078.6 | 3.74 | WIN | 3.89 | WIN | 2.06 | WIN |
| 2-4 | k3_tiny_or | tiny | or | 999 | 0.28/0.57 | 582.3 (plain) | 577.6 | 585.8 | 1615.3 | 1660.1 | 1073.6 | 2.77 | WIN | 2.85 | WIN | 1.86 | WIN |
| 2-4 | k3_medium_or | medium | or | 10,000 | 0.28/0.56 | 557.8 (skip_fused) | 547.4 | 560.1 | 3302.2 | 3247.2 | 1109.5 | 5.92 | WIN | 5.82 | WIN | 2.03 | WIN |
| 2-4 | k3_large_or | large | or | 500,615 | 0.28/0.57 | 529.7 (skip) | 520.3 | 531.2 | 7746.1 | 7615.0 | 1427.3 | 14.62 | WIN | 14.38 | WIN | 2.74 | WIN |
| 2-4 | k3_tiny_mixed | tiny | mixed | 428 | 0.37/0.65 | 547.4 (skip_fused) | 546.1 | 552.2 | 2406.5 | 2385.2 | 1093.4 | 4.40 | WIN | 4.36 | WIN | 2.00 | WIN |
| 2-4 | k3_medium_mixed | medium | mixed | 9,819 | 0.37/0.65 | 536.8 (skip_fused) | 529.3 | 542.7 | 2341.8 | 2395.3 | 1109.9 | 4.36 | WIN | 4.46 | WIN | 2.10 | WIN |
| 2-4 | k3_large_mixed | large | mixed | 499,424 | 0.36/0.65 | 516.1 (skip_fused) | 509.3 | 523.2 | 6041.2 | 6048.1 | 1085.5 | 11.70 | WIN | 11.72 | WIN | 2.13 | WIN |
| 5-16 | k8_tiny_and | tiny | and | 46 | 0.53/2.34 | 959.4 (plain) | 965.7 | 964.9 | 2556.5 | 2602.2 | 1608.9 | 2.66 | WIN | 2.71 | WIN | 1.67 | WIN |
| 5-16 | k8_medium_and | medium | and | 10,097 | 0.52/2.38 | 1362.3 (fused) | 1365.2 | 1378.7 | 3211.5 | 3379.6 | 1232.4 | 2.36 | WIN | 2.48 | WIN | 0.90 | PARITY |
| 5-16 | k8_large_and | large | and | 498,124 | 0.63/2.43 | 1433.0 (fused) | 1425.0 | 1498.7 | 3575.6 | 3563.6 | 1244.8 | 2.50 | WIN | 2.49 | WIN | 0.87 | LOSS |
| 5-16 | k8_tiny_or | tiny | or | 999 | 0.51/2.63 | 1430.3 (skip_fused) | 1422.4 | 1433.4 | 2206.0 | 2246.3 | 1314.8 | 1.54 | WIN | 1.57 | WIN | 0.92 | PARITY |
| 5-16 | k8_medium_or | medium | or | 9,997 | 0.76/4.19 | 1530.3 (fused) | 1484.5 | 1519.6 | 7392.8 | 6884.2 | 1253.4 | 4.83 | WIN | 4.50 | WIN | 0.84 | LOSS |
| 5-16 | k8_large_or | large | or | 498,831 | 0.51/2.61 | 1417.1 (fused) | 1408.1 | 1458.2 | 10587.5 | 10985.8 | 1334.6 | 7.47 | WIN | 7.75 | WIN | 0.95 | PARITY |
| 5-16 | k8_tiny_mixed | tiny | mixed | 91 | 0.87/2.65 | 1429.4 (skip) | 1385.6 | 1394.0 | 6487.3 | 4835.1 | 2416.9 | 4.54 | WIN | 3.38 | WIN | 1.74 | WIN |
| 5-16 | k8_medium_mixed | medium | mixed | 9,901 | 0.86/2.61 | 1399.0 (skip_fused) | 1390.1 | 1440.9 | 10984.4 | 11573.4 | 1398.7 | 7.85 | WIN | 8.27 | WIN | 1.01 | PARITY |
| 5-16 | k8_large_mixed | large | mixed | 500,157 | 0.85/2.62 | 1492.9 (skip_fused) | 1528.4 | 1475.3 | 26765.5 | 27626.3 | 1583.2 | 17.93 | WIN | 18.50 | WIN | 1.04 | PARITY |
| 17+ | k17_tiny_and | tiny | and | 45 | 1.07/7.44 | 2839.3 (skip_fused) | 2888.1 | 2961.9 | 5477.6 | 5055.0 | 2799.7 | 1.93 | WIN | 1.78 | WIN | 0.97 | PARITY |
| 17+ | k17_medium_and | medium | and | 9,754 | 1.06/7.98 | 2909.6 (skip_fused) | 2879.3 | 2873.7 | 7021.7 | 6300.4 | 1972.2 | 2.41 | WIN | 2.17 | WIN | 0.68 | LOSS |
| 17+ | k17_large_and | large | and | 499,819 | 1.06/7.44 | 2776.0 (skip_fused) | 2764.1 | 2786.9 | 9519.3 | 8798.6 | 2216.5 | 3.43 | WIN | 3.17 | WIN | 0.80 | LOSS |
| 17+ | k17_tiny_or | tiny | or | 1,000 | 1.07/7.98 | 2835.1 (skip_fused) | 2854.0 | 2956.5 | 3657.2 | 3620.7 | 2005.9 | 1.29 | WIN | 1.28 | WIN | 0.70 | LOSS |
| 17+ | k17_medium_or | medium | or | 10,000 | 1.04/8.23 | 2853.3 (plain) | 2769.2 | 2996.4 | 9796.8 | 9902.8 | 1761.3 | 3.43 | WIN | 3.47 | WIN | 0.64 | LOSS |
| 17+ | k17_large_or | large | or | 505,114 | 1.06/8.19 | 2798.5 (skip) | 2772.4 | 2785.1 | 17303.5 | 17122.6 | 2216.9 | 6.18 | WIN | 6.12 | WIN | 0.80 | LOSS |
| 17+ | k17_tiny_mixed | tiny | mixed | 46 | 1.91/6.99 | 2378.1 (skip) | 2308.0 | 1620.0 | 7023.1 | 6161.9 | 3655.1 | 2.95 | WIN | 2.59 | WIN | 1.58 | WIN |
| 17+ | k17_medium_mixed | medium | mixed | 10,098 | 2.92/10.53 | 3252.1 (fused) | 3559.2 | 3580.7 | 14138.3 | 14204.1 | 2318.8 | 4.35 | WIN | 4.37 | WIN | 0.65 | LOSS |
| 17+ | k17_large_mixed | large | mixed | 498,269 | 1.91/6.99 | 2838.3 (fused) | 2916.9 | 2881.7 | 49926.1 | 50849.2 | 2620.3 | 17.59 | WIN | 17.92 | WIN | 0.90 | LOSS |
| 17+ | k24_tiny_and | tiny | and | 50 | 1.48/15.64 | 3822.1 (skip_fused) | 3885.5 | 3864.8 | 5905.0 | 5666.5 | 2537.2 | 1.54 | WIN | 1.48 | WIN | 0.65 | LOSS |
| 17+ | k24_medium_and | medium | and | 9,900 | 1.44/15.57 | 3980.4 (skip_fused) | 3775.8 | 3769.6 | 7744.7 | 7598.1 | 2176.2 | 1.95 | WIN | 1.91 | WIN | 0.58 | LOSS |
| 17+ | k24_large_and | large | and | 502,071 | 1.42/15.70 | 4162.7 (skip_fused) | 3867.2 | 3806.4 | 9555.7 | 10058.7 | 1779.0 | 2.30 | WIN | 2.42 | WIN | 0.46 | LOSS |
| 17+ | k24_tiny_or | tiny | or | 1,001 | 1.46/16.35 | 3725.8 (skip) | 3815.8 | 3741.7 | 2949.7 | 2996.9 | 1687.7 | 0.79 | LOSS | 0.80 | LOSS | 0.44 | LOSS |
| 17+ | k24_medium_or | medium | or | 9,995 | 2.18/24.74 | 3713.5 (skip_fused) | 3723.3 | 3736.2 | 9341.3 | 10687.1 | 2293.1 | 2.52 | WIN | 2.88 | WIN | 0.62 | LOSS |
| 17+ | k24_large_or | large | or | 495,651 | 1.44/16.39 | 3749.3 (plain) | 3784.1 | 3719.0 | 21958.3 | 18904.6 | 1900.8 | 5.86 | WIN | 5.04 | WIN | 0.50 | LOSS |
| 17+ | k24_tiny_mixed | tiny | mixed | 48 | 2.62/11.04 | 3872.5 (skip_fused) | 3859.8 | 3999.1 | 26730.1 | 25867.3 | 7325.5 | 6.90 | WIN | 6.68 | WIN | 1.90 | WIN |
| 17+ | k24_medium_mixed | medium | mixed | 9,953 | 2.56/10.94 | 3732.2 (skip_fused) | 3833.5 | 4130.1 | 36972.9 | 38432.2 | 2156.4 | 9.91 | WIN | 10.30 | WIN | 0.56 | LOSS |
| 17+ | k24_large_mixed | large | mixed | 500,471 | 2.56/10.79 | 3885.2 (fused) | 3820.9 | 3818.3 | 60307.6 | 66207.1 | 2320.7 | 15.52 | WIN | 17.04 | WIN | 0.61 | LOSS |
| 2-4 | adv_one_everything | medium | and | 10,000 | 0.21/0.37 | 284.8 (skip) | 295.7 | 301.6 | 1192.5 | 1300.1 | 994.1 | 4.19 | WIN | 4.56 | WIN | 3.36 | WIN |
| 2-4 | adv_one_nothing | medium | and | 1,002 | 0.29/0.50 | 290.0 (plain) | 285.5 | 288.5 | 1669.5 | 1835.7 | 1048.4 | 5.76 | WIN | 6.33 | WIN | 3.67 | WIN |
| 2-4 | adv_or_one_everything | large | or | 1,000,000 | 0.21/0.41 | 333.9 (skip_fused) | 314.0 | 328.3 | 714.9 | - | 802.7 | 2.14 | WIN | - | n/a | 2.56 | WIN |
| 2-4 | adv_or_one_nothing | medium | or | 1,000 | 0.21/0.40 | 316.1 (skip_fused) | 320.7 | 327.7 | 1582.3 | 1662.7 | 974.0 | 5.01 | WIN | 5.26 | WIN | 3.04 | WIN |
| 5-16 | adv_16_nonselective | large | and | 827,152 | 0.96/6.75 | 2742.0 (fused) | 2762.9 | 2774.1 | 7983.7 | 7986.8 | 1664.9 | 2.91 | WIN | 2.91 | WIN | 0.60 | LOSS |
| 5-16 | adv_16_collapse_one | tiny | and | 46 | 0.99/6.59 | 958.5 (skip) | 966.3 | 985.8 | 3516.8 | 3556.4 | 2077.6 | 3.67 | WIN | 3.71 | WIN | 2.15 | WIN |
| 2-4 | adv_correlated_2 | large | and | 500,001 | 0.22/0.38 | 316.3 (fused) | 312.5 | 321.9 | 1566.7 | 1488.7 | 986.1 | 4.95 | WIN | 4.71 | WIN | 3.16 | WIN |
| 5-16 | adv_correlated_5 | large | and | 490,001 | 0.39/1.16 | 612.3 (skip_fused) | 633.6 | 618.3 | 1658.9 | 1662.9 | 1049.8 | 2.71 | WIN | 2.72 | WIN | 1.66 | WIN |
| 2-4 | adv_contradictory_2 | empty | and | 0 | 0.31/0.61 | 271.8 (fused) | 267.4 | 266.5 | 554.7 | 597.6 | 740.4 | 2.04 | WIN | 2.20 | WIN | 2.77 | WIN |
| 17+ | adv_contradictory_17 | empty | and | 0 | 1.04/7.62 | 657.5 (skip) | 665.4 | 697.2 | 1155.3 | 824.1 | 985.5 | 1.76 | WIN | 1.25 | WIN | 1.48 | WIN |
| 2-4 | null_and | large | and | 99,875 | 0.32/0.50 | 346.4 (skip_fused) | 339.5 | 347.6 | 1765.1 | 1708.6 | 963.5 | 5.10 | WIN | 4.93 | WIN | 2.84 | WIN |
| 1 | null_not | large | not | 450,007 | 0.24/0.45 | 186.5 (fused) | 186.3 | 185.5 | 1289.7 | 1252.5 | 845.3 | 6.91 | WIN | 6.71 | WIN | 4.54 | WIN |
| 2-4 | null_or | large | or | 190,006 | 0.36/0.53 | 345.3 (plain) | 356.0 | 352.2 | 3894.8 | 4046.3 | 1001.5 | 11.28 | WIN | 11.72 | WIN | 2.81 | WIN |
| 2-4 | null_not_and | large | not | 850,017 | 0.45/0.85 | 366.1 (skip_fused) | 349.9 | 365.7 | 9881.9 | 9599.8 | 1034.7 | 26.99 | WIN | 26.22 | WIN | 2.96 | WIN |
| 5-16 | null_and_5 | large | and | 267,454 | 0.61/1.50 | 931.4 (skip_fused) | 937.4 | 949.4 | 2810.4 | 2845.1 | 1117.8 | 3.02 | WIN | 3.05 | WIN | 1.19 | WIN |

Representation: Count and Any are `virtual` (no mask survives the call); Rows is `transient` (one Keep mask of 125,000 bytes held for the call); `persistent` never occurs here. Ours rows has no DuckDB counterpart.

Summary, N = 1,000,000, threads = 4 (regular grid only: 39 cases; 15 adversarial/nullable cases excluded from the figures below)

| comparison | weighted ours us | weighted duck us | weighted ratio | weighted | unweighted ours us | unweighted duck us | unweighted ratio | unweighted | geo-mean ratio |
|---|---:|---:|---:|---|---:|---:|---:|---|---:|
| count vs duck count, wall | 692.0 | 4023.8 | 5.81 | WIN | 1997.3 | 10548.1 | 5.28 | WIN | 4.30 |
| count vs duck count, engine | 692.0 | 4051.2 | 5.85 | WIN | 1997.3 | 10629.6 | 5.32 | WIN | 4.27 |
| any vs duck exists, wall | 693.9 | 1202.0 | 1.73 | WIN | 1995.5 | 1851.8 | 0.93 | PARITY | 1.19 |

Per K group (count vs duck count, wall; mean us over regular cases):

| K group | cases | ours us | duck us | ratio | verdict | weight |
|---|---:|---:|---:|---:|---|---:|
| 1 | 3 | 139.9 | 1117.4 | 7.99 | WIN | 0.34 |
| 2-4 | 9 | 544.2 | 3214.4 | 5.91 | WIN | 0.46 |
| 5-16 | 9 | 1383.8 | 8196.3 | 5.92 | WIN | 0.14 |
| 17+ | 18 | 3340.2 | 16962.7 | 5.08 | WIN | 0.06 |

Ours count >= 1000 us (FAILED-GATE on the substrate path, reported): 27 of 54 cases.

