# Evaluation on unseen events

Base model `distilgpt2`, same prompts and seed for both.

| Model | Relevance | Engagement |
|---|---|---|
| Plain distilgpt2 (no adapter) | 0.00 | 0.82 |
| distilgpt2 + merged LoRA | 0.20 | 0.95 |

## Samples from the LoRA model

| Event | Tone | Commentary | Rel. | Eng. |
|---|---|---|---|---|
| Goal by Fernandes, 75th minute, Rovers lead 1-0 | excited | Felix goes over the line and is sent off for 90 minutes! The goalkeeper takes it 100 yards into a final kick at that level! | 0.00 | 1.00 |
| Penalty missed by Walsh, still 0-0 | humorous | That is a penalty miss for Walsh. The referee calls it the kick on its own and that left no one in sight to pay attention. | 0.67 | 0.70 |
| Six by Mehta to bring up his fifty | excited | Seven, seven, eight and nine in the seventy-eighties! Five hundredths of a second now? What are those ten thousand teeth that will make it look like one big toothache! | 0.00 | 1.00 |
| Wicket for Bhatt, caught at slip, 120 for 5 | calm | The wickets are still very loose. A big blunder in the middle and it will be one of those balls that gets thrown away. | 0.00 | 1.00 |
| Slam dunk by Harris to win the game 99-97 | excited | That is it! He blocks that slam, and he pops off his shoes at a very young age. It takes him so long. | 0.33 | 1.00 |
| Double fault by Ivanova on break point | calm | That is it. Ivanovic's half-timbered second and his goalkeeper has missed the third set of a double fault! The final minute remains in good health... | 0.67 | 0.80 |
| Overtake by Silva for second place on lap 40 | excited | Silva takes it one hand and she is in the third. The Brazilian superstar just missed that moment with a huge amount of energy! A superb triple play at this point, thanks to her speed! | 0.33 | 1.00 |
| Overtime goal by Novak, Sharks win 2-1 | excited | That is it! A huge victory in the third period and Snoke has picked up a fantastic one from her own team. | 0.00 | 1.00 |
| Super raid by Deepak, three points | excited | Deepaks snatch the first two and they are at four in the third! They have taken a one point lead with nothing left to do but break away. With just five on board it all! | 0.00 | 1.00 |
| Goalkeeper Brandt drops an easy catch, 1-1 | humorous | The keeper picks the spot and releases it! A huge shocker on that one level with a single header! | 0.00 | 1.00 |
