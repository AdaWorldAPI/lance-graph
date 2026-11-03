# D-GLOBAL-WHERE P2 Part 5: criteria-aware WHERE, N = 10,000,000

Every DuckDB count and EXISTS answer equals the scalar oracle in cases.json (checked before this file was written).

Population targets NOT met by the generator (achieved is what is measured; never assumed from K): k1_tiny_single (target 50, achieved 0), k3_tiny_and (target 50, achieved 0), k3_tiny_or (target 50, achieved 0), k3_tiny_mixed (target 50, achieved 0), k8_tiny_and (target 50, achieved 470), k8_tiny_or (target 50, achieved 0), k8_tiny_mixed (target 50, achieved 935), k17_tiny_and (target 50, achieved 467), k17_tiny_or (target 50, achieved 0), k17_tiny_mixed (target 50, achieved 467), k24_tiny_and (target 50, achieved 472), k24_tiny_or (target 50, achieved 0), k24_tiny_mixed (target 50, achieved 470)

DuckDB build + ANALYZE: threads 1: 5,654,699 us (build 5,653,755, analyze 943); threads 4: 4,333,441 us (build 4,332,828, analyze 613)

### N = 10,000,000, DuckDB threads = 1

| K group | case | pop | shape | achieved | lower plain/fused us | ours count us (order) | ours any us | ours rows us | duck count wall us | duck count engine us | duck exists wall us | count wall ratio | count wall | count engine ratio | count engine | any ratio | any |
|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|---:|---|
| 1 | k1_tiny_single | tiny | single | 0 | 0.10/0.11 | 1688.3 (plain) | 1664.0 | 1707.7 | 477.0 | 592.1 | 591.0 | 0.28 | LOSS | 0.35 | LOSS | 0.36 | LOSS |
| 1 | k1_medium_single | medium | single | 100,002 | 0.10/0.11 | 1684.3 (plain) | 1700.3 | 1751.8 | 16065.5 | 22896.0 | 920.9 | 9.54 | WIN | 13.59 | WIN | 0.54 | LOSS |
| 1 | k1_large_single | large | single | 5,000,008 | 0.09/0.10 | 1748.9 (skip) | 1747.3 | 1759.2 | 14682.0 | 16553.4 | 627.6 | 8.40 | WIN | 9.47 | WIN | 0.36 | LOSS |
| 2-4 | k3_tiny_and | tiny | and | 0 | 0.28/0.50 | 5178.7 (skip_fused) | 5326.2 | 5270.8 | 36607.8 | 37059.9 | 40659.1 | 7.07 | WIN | 7.16 | WIN | 7.63 | WIN |
| 2-4 | k3_medium_and | medium | and | 101,386 | 0.28/0.50 | 5419.4 (fused) | 5422.3 | 5255.3 | 34347.9 | 36076.6 | 699.2 | 6.34 | WIN | 6.66 | WIN | 0.13 | LOSS |
| 2-4 | k3_large_and | large | and | 4,996,187 | 0.29/0.50 | 5251.0 (skip_fused) | 5352.5 | 5727.1 | 41119.5 | 41982.1 | 713.7 | 7.83 | WIN | 8.00 | WIN | 0.13 | LOSS |
| 2-4 | k3_tiny_or | tiny | or | 0 | 0.28/0.59 | 5334.0 (skip_fused) | 5499.0 | 5215.8 | 539.0 | 582.3 | 606.5 | 0.10 | LOSS | 0.11 | LOSS | 0.11 | LOSS |
| 2-4 | k3_medium_or | medium | or | 100,001 | 0.28/0.57 | 5042.6 (fused) | 5064.9 | 5126.4 | 66221.3 | 65374.4 | 1148.4 | 13.13 | WIN | 12.96 | WIN | 0.23 | LOSS |
| 2-4 | k3_large_or | large | or | 5,006,214 | 0.28/0.57 | 5115.4 (skip) | 5159.8 | 5582.6 | 192171.0 | 191457.4 | 813.0 | 37.57 | WIN | 37.43 | WIN | 0.16 | LOSS |
| 2-4 | k3_tiny_mixed | tiny | mixed | 0 | 0.37/0.66 | 5430.3 (skip_fused) | 5656.4 | 5525.2 | 40150.0 | 55077.6 | 52672.7 | 7.39 | WIN | 10.14 | WIN | 9.31 | WIN |
| 2-4 | k3_medium_mixed | medium | mixed | 98,184 | 0.36/0.63 | 5937.1 (skip_fused) | 5703.2 | 5487.3 | 47956.4 | 52098.6 | 791.7 | 8.08 | WIN | 8.78 | WIN | 0.14 | LOSS |
| 2-4 | k3_large_mixed | large | mixed | 4,994,319 | 0.37/0.66 | 5598.2 (plain) | 5345.6 | 5390.8 | 145146.5 | 147578.6 | 816.5 | 25.93 | WIN | 26.36 | WIN | 0.15 | LOSS |
| 5-16 | k8_tiny_and | tiny | and | 470 | 0.74/3.92 | 24039.1 (plain) | 22140.7 | 24092.4 | 54096.7 | 60818.1 | 1617.4 | 2.25 | WIN | 2.53 | WIN | 0.07 | LOSS |
| 5-16 | k8_medium_and | medium | and | 100,924 | 0.51/2.40 | 36402.6 (fused) | 36642.3 | 37327.5 | 83866.3 | 82393.0 | 958.7 | 2.30 | WIN | 2.26 | WIN | 0.03 | LOSS |
| 5-16 | k8_large_and | large | and | 4,981,051 | 0.52/2.39 | 36861.7 (fused) | 36170.7 | 36718.5 | 108897.6 | 106359.7 | 946.6 | 2.95 | WIN | 2.89 | WIN | 0.03 | LOSS |
| 5-16 | k8_tiny_or | tiny | or | 0 | 0.51/2.61 | 36453.4 (skip) | 36412.7 | 36664.2 | 702.6 | 944.2 | 1262.4 | 0.02 | LOSS | 0.03 | LOSS | 0.03 | LOSS |
| 5-16 | k8_medium_or | medium | or | 99,996 | 0.51/2.63 | 36492.9 (skip_fused) | 36093.6 | 36554.4 | 173318.7 | 171537.3 | 957.7 | 4.75 | WIN | 4.70 | WIN | 0.03 | LOSS |
| 5-16 | k8_large_or | large | or | 4,988,280 | 0.51/2.65 | 35830.7 (fused) | 35629.5 | 36135.2 | 310465.8 | 300716.6 | 995.7 | 8.66 | WIN | 8.39 | WIN | 0.03 | LOSS |
| 5-16 | k8_tiny_mixed | tiny | mixed | 935 | 0.88/2.64 | 36491.6 (fused) | 37142.3 | 37749.2 | 150918.1 | 146203.9 | 1770.6 | 4.14 | WIN | 4.01 | WIN | 0.05 | LOSS |
| 5-16 | k8_medium_mixed | medium | mixed | 99,047 | 0.87/2.66 | 36740.6 (skip_fused) | 37822.5 | 37476.0 | 263585.0 | 291324.8 | 1106.1 | 7.17 | WIN | 7.93 | WIN | 0.03 | LOSS |
| 5-16 | k8_large_mixed | large | mixed | 5,001,665 | 0.84/2.56 | 46483.4 (skip_fused) | 47112.3 | 50481.7 | 717301.7 | 687487.8 | 1228.8 | 15.43 | WIN | 14.79 | WIN | 0.03 | LOSS |
| 17+ | k17_tiny_and | tiny | and | 467 | 1.39/18.72 | 74264.2 (skip_fused) | 71629.8 | 71875.5 | 142113.7 | 145205.5 | 2076.0 | 1.91 | WIN | 1.96 | WIN | 0.03 | LOSS |
| 17+ | k17_medium_and | medium | and | 97,561 | 1.06/7.59 | 73592.4 (skip_fused) | 73776.2 | 74798.9 | 159095.2 | 169152.2 | 1784.8 | 2.16 | WIN | 2.30 | WIN | 0.02 | LOSS |
| 17+ | k17_large_and | large | and | 4,998,174 | 1.64/11.69 | 74987.7 (skip_fused) | 73588.6 | 74003.9 | 212825.2 | 222076.1 | 1341.8 | 2.84 | WIN | 2.96 | WIN | 0.02 | LOSS |
| 17+ | k17_tiny_or | tiny | or | 0 | 1.04/8.04 | 74191.9 (skip_fused) | 73149.9 | 76023.9 | 1041.1 | 932.3 | 1473.9 | 0.01 | LOSS | 0.01 | LOSS | 0.02 | LOSS |
| 17+ | k17_medium_or | medium | or | 100,002 | 1.69/12.85 | 73705.7 (skip_fused) | 73464.6 | 73443.5 | 225787.9 | 239839.7 | 1347.3 | 3.06 | WIN | 3.25 | WIN | 0.02 | LOSS |
| 17+ | k17_large_or | large | or | 5,051,202 | 1.01/7.86 | 72316.1 (skip_fused) | 71382.2 | 76371.5 | 478393.1 | 542011.3 | 1349.0 | 6.62 | WIN | 7.50 | WIN | 0.02 | LOSS |
| 17+ | k17_tiny_mixed | tiny | mixed | 467 | 1.89/6.91 | 41805.2 (skip) | 48830.1 | 43736.1 | 136162.5 | 132247.9 | 2176.5 | 3.26 | WIN | 3.16 | WIN | 0.04 | LOSS |
| 17+ | k17_medium_mixed | medium | mixed | 100,925 | 2.76/10.56 | 73473.0 (skip_fused) | 77523.7 | 73476.4 | 293818.0 | 298033.9 | 1420.6 | 4.00 | WIN | 4.06 | WIN | 0.02 | LOSS |
| 17+ | k17_large_mixed | large | mixed | 4,982,809 | 2.96/11.57 | 74632.9 (fused) | 76445.0 | 75979.7 | 1382934.4 | 1401588.6 | 2389.6 | 18.53 | WIN | 18.78 | WIN | 0.03 | LOSS |
| 17+ | k24_tiny_and | tiny | and | 472 | 1.42/15.61 | 93763.9 (fused) | 91951.2 | 88694.3 | 147036.4 | 137556.3 | 1684.9 | 1.57 | WIN | 1.47 | WIN | 0.02 | LOSS |
| 17+ | k24_medium_and | medium | and | 99,057 | 1.90/15.70 | 89354.6 (fused) | 100287.7 | 95396.5 | 188568.6 | 187580.2 | 1508.9 | 2.11 | WIN | 2.10 | WIN | 0.02 | LOSS |
| 17+ | k24_large_and | large | and | 5,020,727 | 2.10/24.46 | 86026.9 (fused) | 85642.2 | 90768.2 | 248329.8 | 243513.3 | 1504.5 | 2.89 | WIN | 2.83 | WIN | 0.02 | LOSS |
| 17+ | k24_tiny_or | tiny | or | 0 | 1.46/16.60 | 88091.3 (plain) | 87496.3 | 87911.4 | 1002.4 | 1046.2 | 1129.8 | 0.01 | LOSS | 0.01 | LOSS | 0.01 | LOSS |
| 17+ | k24_medium_or | medium | or | 99,996 | 1.45/16.55 | 90460.6 (skip) | 86339.9 | 88804.6 | 227149.9 | 239568.3 | 1463.5 | 2.51 | WIN | 2.65 | WIN | 0.02 | LOSS |
| 17+ | k24_large_or | large | or | 4,956,413 | 2.26/25.33 | 89297.0 (skip) | 86766.6 | 86445.5 | 572694.7 | 558926.0 | 1600.6 | 6.41 | WIN | 6.26 | WIN | 0.02 | LOSS |
| 17+ | k24_tiny_mixed | tiny | mixed | 470 | 3.57/16.52 | 90773.9 (fused) | 90343.5 | 88104.2 | 657116.1 | 567573.7 | 1916.5 | 7.24 | WIN | 6.25 | WIN | 0.02 | LOSS |
| 17+ | k24_medium_mixed | medium | mixed | 99,512 | 2.60/11.12 | 89640.1 (fused) | 84966.5 | 86376.5 | 932183.2 | 981755.7 | 1784.2 | 10.40 | WIN | 10.95 | WIN | 0.02 | LOSS |
| 17+ | k24_large_mixed | large | mixed | 5,004,671 | 2.58/11.12 | 85767.4 (fused) | 85184.7 | 85509.0 | 2090124.5 | 1985161.1 | 2006.5 | 24.37 | WIN | 23.15 | WIN | 0.02 | LOSS |
| 2-4 | adv_one_everything | medium | and | 100,002 | 0.28/0.63 | 3643.2 (skip_fused) | 3536.2 | 3374.6 | 14886.8 | 24485.8 | 972.8 | 4.09 | WIN | 6.72 | WIN | 0.28 | LOSS |
| 2-4 | adv_one_nothing | medium | and | 10,002 | 0.40/0.79 | 2803.7 (skip) | 4055.7 | 2853.7 | 25638.5 | 29261.4 | 1090.4 | 9.14 | WIN | 10.44 | WIN | 0.27 | LOSS |
| 2-4 | adv_or_one_everything | large | or | 10,000,000 | 0.21/0.39 | 3465.4 (skip) | 3458.5 | 3461.7 | 847.4 | - | 922.9 | 0.24 | LOSS | - | n/a | 0.27 | LOSS |
| 2-4 | adv_or_one_nothing | medium | or | 10,000 | 0.21/0.40 | 3429.9 (skip) | 3475.6 | 3595.3 | 22805.1 | 23057.3 | 713.4 | 6.65 | WIN | 6.72 | WIN | 0.21 | LOSS |
| 5-16 | adv_16_nonselective | large | and | 8,271,472 | 0.96/6.74 | 72430.2 (fused) | 71969.7 | 70735.1 | 217536.6 | 234985.9 | 1441.2 | 3.00 | WIN | 3.24 | WIN | 0.02 | LOSS |
| 5-16 | adv_16_collapse_one | medium | and | 472 | 0.97/6.64 | 19888.7 (plain) | 20468.5 | 20370.5 | 64305.0 | 64808.0 | 1413.1 | 3.23 | WIN | 3.26 | WIN | 0.07 | LOSS |
| 2-4 | adv_correlated_2 | large | and | 5,000,004 | 0.22/0.37 | 3531.9 (fused) | 3495.6 | 3595.0 | 28390.5 | 30240.1 | 668.0 | 8.04 | WIN | 8.56 | WIN | 0.19 | LOSS |
| 5-16 | adv_correlated_5 | large | and | 4,900,007 | 0.55/1.15 | 12279.7 (fused) | 13044.9 | 12683.7 | 31018.7 | 32148.3 | 782.3 | 2.53 | WIN | 2.62 | WIN | 0.06 | LOSS |
| 2-4 | adv_contradictory_2 | empty | and | 0 | 0.21/0.37 | 2926.0 (skip_fused) | 2904.2 | 2977.3 | 727.8 | 683.2 | 856.1 | 0.25 | LOSS | 0.23 | LOSS | 0.29 | LOSS |
| 17+ | adv_contradictory_17 | empty | and | 0 | 1.06/7.83 | 7505.2 (plain) | 8087.9 | 14375.9 | 711.0 | 789.9 | 821.3 | 0.09 | LOSS | 0.11 | LOSS | 0.10 | LOSS |
| 2-4 | null_and | large | and | 999,196 | 0.32/0.49 | 4670.4 (skip_fused) | 5726.6 | 4810.2 | 36063.3 | 36621.3 | 1035.5 | 7.72 | WIN | 7.84 | WIN | 0.18 | LOSS |
| 1 | null_not | large | not | 4,499,990 | 0.38/0.76 | 2179.7 (skip_fused) | 2179.2 | 2221.7 | 20619.5 | 20156.2 | 775.5 | 9.46 | WIN | 9.25 | WIN | 0.36 | LOSS |
| 2-4 | null_or | large | or | 1,900,011 | 0.32/0.53 | 4684.8 (skip) | 4032.7 | 4904.5 | 99873.4 | 132234.7 | 1117.0 | 21.32 | WIN | 28.23 | WIN | 0.28 | LOSS |
| 2-4 | null_not_and | large | not | 8,500,703 | 0.44/0.87 | 3955.8 (fused) | 3971.0 | 4159.1 | 268063.0 | 316992.9 | 787.3 | 67.76 | WIN | 80.13 | WIN | 0.20 | LOSS |
| 5-16 | null_and_5 | large | and | 2,674,492 | 0.92/2.13 | 21901.5 (fused) | 22161.4 | 22509.1 | 70334.6 | 73451.0 | 814.6 | 3.21 | WIN | 3.35 | WIN | 0.04 | LOSS |

Representation: Count and Any are `virtual` (no mask survives the call); Rows is `transient` (one Keep mask of 1,250,000 bytes held for the call); `persistent` never occurs here. Ours rows has no DuckDB counterpart.

Summary, N = 10,000,000, threads = 1 (regular grid only: 39 cases; 15 adversarial/nullable cases excluded from the figures below)

| comparison | weighted ours us | weighted duck us | weighted ratio | weighted | unweighted ours us | unweighted duck us | unweighted ratio | unweighted | geo-mean ratio |
|---|---:|---:|---:|---|---:|---:|---:|---|---:|
| count vs duck count, wall | 12904.5 | 90393.4 | 7.00 | WIN | 46547.9 | 271667.0 | 5.84 | WIN | 3.06 |
| count vs duck count, engine | 12904.5 | 92188.6 | 7.14 | WIN | 46547.9 | 270997.0 | 5.82 | WIN | 3.19 |
| any vs duck exists, wall | 12913.8 | 5567.0 | 0.43 | LOSS | 46604.5 | 3637.5 | 0.08 | LOSS | 0.06 |

Per K group (count vs duck count, wall; mean us over regular cases):

| K group | cases | ours us | duck us | ratio | verdict | weight |
|---|---:|---:|---:|---:|---|---:|
| 1 | 3 | 1707.2 | 10408.2 | 6.10 | WIN | 0.34 |
| 2-4 | 9 | 5367.4 | 67139.9 | 12.51 | WIN | 0.46 |
| 5-16 | 9 | 36199.5 | 207017.0 | 5.72 | WIN | 0.14 |
| 17+ | 18 | 79785.8 | 449798.7 | 5.64 | WIN | 0.06 |

Ours count >= 1000 us (FAILED-GATE on the substrate path, reported): 54 of 54 cases.

### N = 10,000,000, DuckDB threads = 4

| K group | case | pop | shape | achieved | lower plain/fused us | ours count us (order) | ours any us | ours rows us | duck count wall us | duck count engine us | duck exists wall us | count wall ratio | count wall | count engine ratio | count engine | any ratio | any |
|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|---:|---|
| 1 | k1_tiny_single | tiny | single | 0 | 0.10/0.11 | 1688.3 (plain) | 1664.0 | 1707.7 | 749.2 | 608.7 | 740.3 | 0.44 | LOSS | 0.36 | LOSS | 0.44 | LOSS |
| 1 | k1_medium_single | medium | single | 100,002 | 0.10/0.11 | 1684.3 (plain) | 1700.3 | 1751.8 | 4115.6 | 4174.6 | 878.1 | 2.44 | WIN | 2.48 | WIN | 0.52 | LOSS |
| 1 | k1_large_single | large | single | 5,000,008 | 0.09/0.10 | 1748.9 (skip) | 1747.3 | 1759.2 | 4483.3 | 4355.8 | 913.8 | 2.56 | WIN | 2.49 | WIN | 0.52 | LOSS |
| 2-4 | k3_tiny_and | tiny | and | 0 | 0.28/0.50 | 5178.7 (skip_fused) | 5326.2 | 5270.8 | 10953.8 | 10635.1 | 10197.0 | 2.12 | WIN | 2.05 | WIN | 1.91 | WIN |
| 2-4 | k3_medium_and | medium | and | 101,386 | 0.28/0.50 | 5419.4 (fused) | 5422.3 | 5255.3 | 9979.6 | 10728.6 | 1034.5 | 1.84 | WIN | 1.98 | WIN | 0.19 | LOSS |
| 2-4 | k3_large_and | large | and | 4,996,187 | 0.29/0.50 | 5251.0 (skip_fused) | 5352.5 | 5727.1 | 12053.1 | 12181.6 | 1018.2 | 2.30 | WIN | 2.32 | WIN | 0.19 | LOSS |
| 2-4 | k3_tiny_or | tiny | or | 0 | 0.28/0.59 | 5334.0 (skip_fused) | 5499.0 | 5215.8 | 648.2 | 678.1 | 750.1 | 0.12 | LOSS | 0.13 | LOSS | 0.14 | LOSS |
| 2-4 | k3_medium_or | medium | or | 100,001 | 0.28/0.57 | 5042.6 (fused) | 5064.9 | 5126.4 | 18505.3 | 18795.1 | 1011.7 | 3.67 | WIN | 3.73 | WIN | 0.20 | LOSS |
| 2-4 | k3_large_or | large | or | 5,006,214 | 0.28/0.57 | 5115.4 (skip) | 5159.8 | 5582.6 | 50145.2 | 54685.3 | 1035.6 | 9.80 | WIN | 10.69 | WIN | 0.20 | LOSS |
| 2-4 | k3_tiny_mixed | tiny | mixed | 0 | 0.37/0.66 | 5430.3 (skip_fused) | 5656.4 | 5525.2 | 10858.1 | 10898.2 | 11380.8 | 2.00 | WIN | 2.01 | WIN | 2.01 | WIN |
| 2-4 | k3_medium_mixed | medium | mixed | 98,184 | 0.36/0.63 | 5937.1 (skip_fused) | 5703.2 | 5487.3 | 13830.3 | 12542.5 | 976.2 | 2.33 | WIN | 2.11 | WIN | 0.17 | LOSS |
| 2-4 | k3_large_mixed | large | mixed | 4,994,319 | 0.37/0.66 | 5598.2 (plain) | 5345.6 | 5390.8 | 37466.7 | 40296.9 | 1052.1 | 6.69 | WIN | 7.20 | WIN | 0.20 | LOSS |
| 5-16 | k8_tiny_and | tiny | and | 470 | 0.74/3.92 | 24039.1 (plain) | 22140.7 | 24092.4 | 16602.2 | 19904.8 | 1921.7 | 0.69 | LOSS | 0.83 | LOSS | 0.09 | LOSS |
| 5-16 | k8_medium_and | medium | and | 100,924 | 0.51/2.40 | 36402.6 (fused) | 36642.3 | 37327.5 | 26988.8 | 24283.4 | 1249.7 | 0.74 | LOSS | 0.67 | LOSS | 0.03 | LOSS |
| 5-16 | k8_large_and | large | and | 4,981,051 | 0.52/2.39 | 36861.7 (fused) | 36170.7 | 36718.5 | 30166.4 | 29397.2 | 1241.5 | 0.82 | LOSS | 0.80 | LOSS | 0.03 | LOSS |
| 5-16 | k8_tiny_or | tiny | or | 0 | 0.51/2.61 | 36453.4 (skip) | 36412.7 | 36664.2 | 798.1 | 851.8 | 1022.5 | 0.02 | LOSS | 0.02 | LOSS | 0.03 | LOSS |
| 5-16 | k8_medium_or | medium | or | 99,996 | 0.51/2.63 | 36492.9 (skip_fused) | 36093.6 | 36554.4 | 51723.3 | 46573.0 | 1226.4 | 1.42 | WIN | 1.28 | WIN | 0.03 | LOSS |
| 5-16 | k8_large_or | large | or | 4,988,280 | 0.51/2.65 | 35830.7 (fused) | 35629.5 | 36135.2 | 84262.8 | 89431.8 | 1399.6 | 2.35 | WIN | 2.50 | WIN | 0.04 | LOSS |
| 5-16 | k8_tiny_mixed | tiny | mixed | 935 | 0.88/2.64 | 36491.6 (fused) | 37142.3 | 37749.2 | 43857.1 | 43639.9 | 2757.2 | 1.20 | WIN | 1.20 | WIN | 0.07 | LOSS |
| 5-16 | k8_medium_mixed | medium | mixed | 99,047 | 0.87/2.66 | 36740.6 (skip_fused) | 37822.5 | 37476.0 | 78663.7 | 87577.5 | 1337.5 | 2.14 | WIN | 2.38 | WIN | 0.04 | LOSS |
| 5-16 | k8_large_mixed | large | mixed | 5,001,665 | 0.84/2.56 | 46483.4 (skip_fused) | 47112.3 | 50481.7 | 187067.9 | 192926.0 | 1470.6 | 4.02 | WIN | 4.15 | WIN | 0.03 | LOSS |
| 17+ | k17_tiny_and | tiny | and | 467 | 1.39/18.72 | 74264.2 (skip_fused) | 71629.8 | 71875.5 | 41400.9 | 37266.6 | 2457.8 | 0.56 | LOSS | 0.50 | LOSS | 0.03 | LOSS |
| 17+ | k17_medium_and | medium | and | 97,561 | 1.06/7.59 | 73592.4 (skip_fused) | 73776.2 | 74798.9 | 43895.3 | 45147.6 | 1538.0 | 0.60 | LOSS | 0.61 | LOSS | 0.02 | LOSS |
| 17+ | k17_large_and | large | and | 4,998,174 | 1.64/11.69 | 74987.7 (skip_fused) | 73588.6 | 74003.9 | 57312.3 | 55350.1 | 1651.6 | 0.76 | LOSS | 0.74 | LOSS | 0.02 | LOSS |
| 17+ | k17_tiny_or | tiny | or | 0 | 1.04/8.04 | 74191.9 (skip_fused) | 73149.9 | 76023.9 | 1239.1 | 1011.0 | 1299.2 | 0.02 | LOSS | 0.01 | LOSS | 0.02 | LOSS |
| 17+ | k17_medium_or | medium | or | 100,002 | 1.69/12.85 | 73705.7 (skip_fused) | 73464.6 | 73443.5 | 70731.7 | 71277.0 | 1534.5 | 0.96 | PARITY | 0.97 | PARITY | 0.02 | LOSS |
| 17+ | k17_large_or | large | or | 5,051,202 | 1.01/7.86 | 72316.1 (skip_fused) | 71382.2 | 76371.5 | 132258.5 | 128086.9 | 1600.5 | 1.83 | WIN | 1.77 | WIN | 0.02 | LOSS |
| 17+ | k17_tiny_mixed | tiny | mixed | 467 | 1.89/6.91 | 41805.2 (skip) | 48830.1 | 43736.1 | 37984.4 | 42628.1 | 3366.9 | 0.91 | PARITY | 1.02 | PARITY | 0.07 | LOSS |
| 17+ | k17_medium_mixed | medium | mixed | 100,925 | 2.76/10.56 | 73473.0 (skip_fused) | 77523.7 | 73476.4 | 82346.8 | 85098.5 | 1813.5 | 1.12 | WIN | 1.16 | WIN | 0.02 | LOSS |
| 17+ | k17_large_mixed | large | mixed | 4,982,809 | 2.96/11.57 | 74632.9 (fused) | 76445.0 | 75979.7 | 363912.1 | 367205.7 | 2557.8 | 4.88 | WIN | 4.92 | WIN | 0.03 | LOSS |
| 17+ | k24_tiny_and | tiny | and | 472 | 1.42/15.61 | 93763.9 (fused) | 91951.2 | 88694.3 | 42430.7 | 53442.5 | 2717.2 | 0.45 | LOSS | 0.57 | LOSS | 0.03 | LOSS |
| 17+ | k24_medium_and | medium | and | 99,057 | 1.90/15.70 | 89354.6 (fused) | 100287.7 | 95396.5 | 55181.2 | 65205.9 | 2579.0 | 0.62 | LOSS | 0.73 | LOSS | 0.03 | LOSS |
| 17+ | k24_large_and | large | and | 5,020,727 | 2.10/24.46 | 86026.9 (fused) | 85642.2 | 90768.2 | 68975.0 | 69978.9 | 2407.8 | 0.80 | LOSS | 0.81 | LOSS | 0.03 | LOSS |
| 17+ | k24_tiny_or | tiny | or | 0 | 1.46/16.60 | 88091.3 (plain) | 87496.3 | 87911.4 | 1637.8 | 1171.0 | 1724.2 | 0.02 | LOSS | 0.01 | LOSS | 0.02 | LOSS |
| 17+ | k24_medium_or | medium | or | 99,996 | 1.45/16.55 | 90460.6 (skip) | 86339.9 | 88804.6 | 88748.3 | 104740.6 | 3027.8 | 0.98 | PARITY | 1.16 | WIN | 0.04 | LOSS |
| 17+ | k24_large_or | large | or | 4,956,413 | 2.26/25.33 | 89297.0 (skip) | 86766.6 | 86445.5 | 215605.7 | 212426.9 | 3283.8 | 2.41 | WIN | 2.38 | WIN | 0.04 | LOSS |
| 17+ | k24_tiny_mixed | tiny | mixed | 470 | 3.57/16.52 | 90773.9 (fused) | 90343.5 | 88104.2 | 183991.0 | 179896.0 | 7625.6 | 2.03 | WIN | 1.98 | WIN | 0.08 | LOSS |
| 17+ | k24_medium_mixed | medium | mixed | 99,512 | 2.60/11.12 | 89640.1 (fused) | 84966.5 | 86376.5 | 268751.0 | 291474.1 | 2621.0 | 3.00 | WIN | 3.25 | WIN | 0.03 | LOSS |
| 17+ | k24_large_mixed | large | mixed | 5,004,671 | 2.58/11.12 | 85767.4 (fused) | 85184.7 | 85509.0 | 556772.8 | 516854.5 | 2286.9 | 6.49 | WIN | 6.03 | WIN | 0.03 | LOSS |
| 2-4 | adv_one_everything | medium | and | 100,002 | 0.28/0.63 | 3643.2 (skip_fused) | 3536.2 | 3374.6 | 3725.4 | 4034.9 | 901.8 | 1.02 | PARITY | 1.11 | WIN | 0.26 | LOSS |
| 2-4 | adv_one_nothing | medium | and | 10,002 | 0.40/0.79 | 2803.7 (skip) | 4055.7 | 2853.7 | 7062.0 | 7261.5 | 1009.7 | 2.52 | WIN | 2.59 | WIN | 0.25 | LOSS |
| 2-4 | adv_or_one_everything | large | or | 10,000,000 | 0.21/0.39 | 3465.4 (skip) | 3458.5 | 3461.7 | 633.2 | - | 771.4 | 0.18 | LOSS | - | n/a | 0.22 | LOSS |
| 2-4 | adv_or_one_nothing | medium | or | 10,000 | 0.21/0.40 | 3429.9 (skip) | 3475.6 | 3595.3 | 5287.0 | 5569.3 | 916.4 | 1.54 | WIN | 1.62 | WIN | 0.26 | LOSS |
| 5-16 | adv_16_nonselective | large | and | 8,271,472 | 0.96/6.74 | 72430.2 (fused) | 71969.7 | 70735.1 | 57095.4 | 59265.9 | 1649.1 | 0.79 | LOSS | 0.82 | LOSS | 0.02 | LOSS |
| 5-16 | adv_16_collapse_one | medium | and | 472 | 0.97/6.64 | 19888.7 (plain) | 20468.5 | 20370.5 | 18712.3 | 19949.1 | 2643.1 | 0.94 | PARITY | 1.00 | PARITY | 0.13 | LOSS |
| 2-4 | adv_correlated_2 | large | and | 5,000,004 | 0.22/0.37 | 3531.9 (fused) | 3495.6 | 3595.0 | 7181.1 | 6992.3 | 921.0 | 2.03 | WIN | 1.98 | WIN | 0.26 | LOSS |
| 5-16 | adv_correlated_5 | large | and | 4,900,007 | 0.55/1.15 | 12279.7 (fused) | 13044.9 | 12683.7 | 6878.6 | 6917.5 | 949.2 | 0.56 | LOSS | 0.56 | LOSS | 0.07 | LOSS |
| 2-4 | adv_contradictory_2 | empty | and | 0 | 0.21/0.37 | 2926.0 (skip_fused) | 2904.2 | 2977.3 | 656.9 | 676.6 | 947.9 | 0.22 | LOSS | 0.23 | LOSS | 0.33 | LOSS |
| 17+ | adv_contradictory_17 | empty | and | 0 | 1.06/7.83 | 7505.2 (plain) | 8087.9 | 14375.9 | 790.0 | 882.2 | 949.5 | 0.11 | LOSS | 0.12 | LOSS | 0.12 | LOSS |
| 2-4 | null_and | large | and | 999,196 | 0.32/0.49 | 4670.4 (skip_fused) | 5726.6 | 4810.2 | 8346.3 | 8109.7 | 961.5 | 1.79 | WIN | 1.74 | WIN | 0.17 | LOSS |
| 1 | null_not | large | not | 4,499,990 | 0.38/0.76 | 2179.7 (skip_fused) | 2179.2 | 2221.7 | 5210.3 | 5399.1 | 1018.5 | 2.39 | WIN | 2.48 | WIN | 0.47 | LOSS |
| 2-4 | null_or | large | or | 1,900,011 | 0.32/0.53 | 4684.8 (skip) | 4032.7 | 4904.5 | 26963.6 | 27898.9 | 1347.2 | 5.76 | WIN | 5.96 | WIN | 0.33 | LOSS |
| 2-4 | null_not_and | large | not | 8,500,703 | 0.44/0.87 | 3955.8 (fused) | 3971.0 | 4159.1 | 74116.9 | 74124.0 | 1188.5 | 18.74 | WIN | 18.74 | WIN | 0.30 | LOSS |
| 5-16 | null_and_5 | large | and | 2,674,492 | 0.92/2.13 | 21901.5 (fused) | 22161.4 | 22509.1 | 17936.4 | 17825.6 | 1060.5 | 0.82 | LOSS | 0.81 | LOSS | 0.05 | LOSS |

Representation: Count and Any are `virtual` (no mask survives the call); Rows is `transient` (one Keep mask of 1,250,000 bytes held for the call); `persistent` never occurs here. Ours rows has no DuckDB counterpart.

Summary, N = 10,000,000, threads = 4 (regular grid only: 39 cases; 15 adversarial/nullable cases excluded from the figures below)

| comparison | weighted ours us | weighted duck us | weighted ratio | weighted | unweighted ours us | unweighted duck us | unweighted ratio | unweighted | geo-mean ratio |
|---|---:|---:|---:|---|---:|---:|---:|---|---:|
| count vs duck count, wall | 12904.5 | 25265.7 | 1.96 | WIN | 46547.9 | 77105.0 | 1.66 | WIN | 1.07 |
| count vs duck count, engine | 12904.5 | 25875.0 | 2.01 | WIN | 46547.9 | 78036.6 | 1.68 | WIN | 1.08 |
| any vs duck exists, wall | 12913.8 | 2107.0 | 0.16 | LOSS | 46604.5 | 2325.8 | 0.05 | LOSS | 0.07 |

Per K group (count vs duck count, wall; mean us over regular cases):

| K group | cases | ours us | duck us | ratio | verdict | weight |
|---|---:|---:|---:|---:|---|---:|
| 1 | 3 | 1707.2 | 3116.0 | 1.83 | WIN | 0.34 |
| 2-4 | 9 | 5367.4 | 18271.1 | 3.40 | WIN | 0.46 |
| 5-16 | 9 | 36199.5 | 57792.3 | 1.60 | WIN | 0.14 |
| 17+ | 18 | 79785.8 | 128509.7 | 1.61 | WIN | 0.06 |

Ours count >= 1000 us (FAILED-GATE on the substrate path, reported): 54 of 54 cases.

