# Reading the analysis

- Times are positions in the original video; click a time to play from that moment.
- Confidence is the model's probability for the tactic. Coordinated Attack and Coordinated Defense calls above 50% were right about 85% of the time on videos the model never saw. Delayed Support and Spacing Breakdown are less reliable (about 1 in 4) and should be checked by eye.
- Players are named by tracker number (Player #7). The number is not the shirt number. Use the player's picture to recognise them. The tracker can occasionally swap numbers when two players cross.
- Impact: +1 when a player drives an attack, +0.5 for each defence they were part of, -1 for a late reaction or a spacing error, -0.5 when out of step in a defence; shown per minute on court.
- Distance and speed come from court positions in metres; short tracking glitches are ignored.
