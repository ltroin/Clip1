*Table: The attack success rate on various checkpoints for the Vicuna model was evaluated using the Clip method. Specifically, for Length 1, the continuous version was assessed, whereas for Lengths 40 and 100, the hybrid version was examined. This evaluation aims to represent general cases.*
| Length | Alpha | ASR@100 | w/o Clip | ASR@500 | w/o Clip | ASR@1000 | w/o Clip |
|--------|-------|---------|----------|---------|----------|----------|----------|
| 1      | 5     | 0%      | 62%      | 2%      | **85%**  | 7%       | **87%**  |
|        | 7     | 8%      |          | 28%     |          | 35%      |          |
|        | 10    | 47%     |          | 68%     |          | 68%      |          |
|        | 20    | **67%** |          | 75%     |          | 77%      |          |
| 40     | 5     | 78%     | 75%      | 78%     | 67%      | 80%      | 62%      |
|        | 7     | **95%** |          | **85%** |          | **83%**  |          |
|        | 10    | 82%     |          | 75%     |          | 77%      |          |
| 100    | 5     | **78%** | 72%      | **77%** | 47%      | **83%**  | 38%      |
|        | 7     | 72%     |          | 75%     |          | 67%      |          |
|        | 10    | 67%     |          | 62%     |          | 67%      |          |

