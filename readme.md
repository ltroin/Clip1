# Continuous Embedding Attacks via Clipped Inputs in Jailbreaking Large Language Models

## Running the Code

### Without the Clip Method
To run the attack without using the Clip method:
```bash
python attack.py --mode mixed --task no
```

### With the Clip Method
To run the attack using the Clip method:

```bash
python attack.py --mode mixed --task sigma
```

## Evaluating the Results
Modify the code in test.py as follows based on the task type:

- For tasks without the Clip method, remove the sigma keyword from lines 194 and 198.
- For tasks with the Clip method, ensure the sigma keyword is included in lines 194 and 198.
After making the necessary modifications, evaluate the results by running:
```bash
python test.py
```
