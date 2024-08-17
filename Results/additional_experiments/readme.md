## Normal Distribution without CLIP

As expected, this resulted in almost no successful attack cases for length 40 and length 100. However, for length 1, more cases are successful, which supports the conclusion that a small length is a strong regularizer. We will furthermore include this additional evidence in our paper.

| Length | 100  | 500  | 1000 |
|--------|------|------|------|
| 1      | 0.32 | 0.48 | 0.47 |
| 40     | 0.03 | 0.02 | 0    |
| 100    | 0.02 | 0.02 | 0    |



## Normal Distribution with CLIP: 

We employed the continuous version for length 1 and the hybrid version for lengths 40 and 100. Our results indicate that the performance sometimes surpasses that of Table 2 (Proper Initial Distribution with CLIP) (PDC) and sometimes does not. This suggests that, as long as the CLIP method is used to clip the input, the initialization has minimal impact. This is because, even if the input is not initialized around the mean, it will be projected around the mean in subsequent iterations due to the CLIP method. Therefore, the primary difference lies in whether the initial input is sampled around the mean of the vocabulary. We will include this observation in our paper.


<!-- | Length | Alpha | ASR@100 | ASR@500 | ASR@1000 |
|--------|-------|---------|---------|----------|
| 1      | 5     | 0.12    | 0.45    | 0.43     |
|        | 7     | 0.33    | 0.78    | 0.72     |
|        | 10    | 0.48    | **0.95** | 0.82    |
|        | 20    | **0.57** | **0.95** | **0.78** |
| 40     | 5     | 0.68    | 0.77    | 0.77     |
|        | 7     | 0.88    | 0.65    | 0.68     |
|        | 10    | **0.90** | **0.77** | **0.77** |
| 100    | 5     | 0.62    | 0.68    | 0.70     |
|        | 7     | 0.75    | 0.60    | 0.60     |
|        | 10    | **0.83** | **0.68** | **0.70** | -->
| Length | Alpha | **ASR@100** | **PDC@100** | **ASR@500** | **PDC@500** | **ASR@1000** | **PDC@1000** |
|--------|-------|-------------|-------------|-------------|-------------|--------------|--------------|
| 1      | 5     | **12%**     | 5%          | **45%**     | 18%         | **43%**      | 8%           |
| 1      | 7     | **33%**     | 32%         | **78%**     | 52%         | 72%          | **63%**      |
| 1      | 10    | 48%         | **63%**     | **95%**     | 73%         | 82%          | **85%**      |
| 1      | 20    | 57%         | **73%**     | 95%         | **90%**     | 78%          | **95%**      |
| 40     | 5     | 68%         | **82%**     | 77%         | **87%**     | 77%          | **83%**      |
| 40     | 7     | **88%**     | 83%         | 65%         | **82%**     | 68%          | **82%**      |
| 40     | 10    | **90%**     | 82%         | **77%**     | 58%         | **77%**      | 62%          |
| 100    | 5     | 62%         | **70%**     | 68%         | **58%**     | **70%**      | 60%          |
| 100    | 7     | **75%**     | 65%         | 60%         | **63%**     | **60%**      | 60%          |
| 100    | 10    | **83%**     | 67%         | 68%         | **47%**     | **70%**      | 45%          |



## Comparison of ASR Metrics for Different Attack Methods
We conducted preliminary experiments to bridge this gap in our study. Specifically, we followed the original parameters from the papers by Zou et al. (2023b) and Schwinn et al. (2023), utilized a random sample of five cases from our benchmark (see Data/data_new.csv), and observed the following differences:

### Efficiency:

Discrete Prefix optimization (Zou et al., 2023b) often leads to a higher number of iterations and a longer time for a successful jailbreak (1,616 iterations, 4.7 hours for these five samples). Additionally, we observed that the LLama-2 7b model is more difficult to attack than Vicuna. However, for Continuous Prefix (Schwinn et al., 2023) and Continuous Input (our study, using the best parameters from our paper), the number of iterations and time consumption are much lower than with Discrete Prefix (Schwinn: 47 iterations, 5.9 minutes; ours: 87 iterations, 5.9 minutes) and there is no observable difference in jailbreak difficulty between LLama-2 7b and Vicuna. This reflects that continuous attacks have a better ability to exploit loopholes. Our method also tests more cases in the same time period.

### ASR:
Continuous Prefix (Schwinn et al., 2023) and our method both achieved a 100% success rate on the selected benchmark, whereas Discrete Prefix (Zou et al., 2023b) achieved only 40%.

### Transferability:
Discrete Prefix (Zou et al., 2023b) has better transferability due to the discrete tokens' nature of the input attack. This raises more concerns and potential societal threats than the other two methods.


| Attack Method       | ASR Metric |
| ------------------- |:----------:|
| Continuous Input    | 100%       |
| Continuous Suffix   | 100%       |
| Discrete Suffix     | 40%        |

