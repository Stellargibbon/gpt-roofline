## Card 1 - the contract

This is a predictions file written before run #3 and will not be touched again to ensure it was written before and not tampered with.

## Card 2 - selection gap

The best-val optimism for a random-batch-eval was measured once during run #1 which came out to 0.0197. The eval-batch noise had a sd of 0.0064 (spread [4.2148, 4.2342]), I predict that in any upcoming run using the random-batch-eval recipe, the band will be between 0.015 and 0.025 since everything was constant except randomized vs set-seeded runs and run 1 was trained on a RTX PRO 6000 vs run 2's GB10
tail = 100, sd = fresh draws, count = raw n not n_eff

## Card 3 - mechanism claim

I'm predicting that the fixed-eval-set recipe will reduce the gap to essentially 0[0.0115 of 0] with a tiny bit of eval noise from the frozen eval set. I'm picking this band so a tiny bit of noise is acceptable but not to the magnitude of 0.0197.

## Card 4 - seed spread

If nothing changes inbetween 2 runs except for a RUN_SEED, at n seeds I expect the measured sd to land in [0.002, 0.010]. Currently, the one datapoint I have is sd = 0.0047 (this anchor is approximate [0.003-0.005 depending on how a run's landing number is defined]) from run #2s A and B run, which isn't a lot of information to go off of. So I'm giving it a range of half to double the current sd.

## Card 5 - minimum detectable effect

I am setting a rule that a difference only counts if it beats the floor that the variance sets. Using the rule of thumb that a detectable d needs d > ~2*sd*sqrt(2/n), I can plug in my sd = 0.0047, n=1 and estimate 2*0.0047*sqrt(2/1) = ~0.013 (this is an estimate since I'm assuming sd is exactly 0.0047). If the difference falls under the number, it's declared as a tie with no better run. Using run #2 A and B as an example, run 2A (seed 2): 4.1164, run 2B (seed 1): 4.1205, difference: ~0.0041, which sits below the rule's ~0.013 floor

## Card 6 - falsifier ledger

Card 2 fails if a random-batch-eval run's gap lands outside [0.015, 0.025].
Card 3 fails if a fixed-eval-set run's gap exceeds 0.0115 of 0
Card 4 fails if the measured between-seed sd lands outside [0.002, 0.010].
