# //// pandera
*IMC Prosperity 3 - Our first time taking a swing away from the books*

Closely following the footsteps of other competition takers, we had also decided to do a write up with the goal of reflecting on what we could’ve improved upon for our next competition. 

Pandera was a team consisting of Shiv Mehta, Alex Zhang, Jayendra Mangal and Yash Tandon. I (Alex) had met Shiv and friends through doing Jane Street puzzles sometime last year, and bonded over our shared interest in data science and finance. Unfortunately, we could only finish up to round 3 before taking our leave, however, this write up serves to solidify the skills and knowledge we picked up along the way, as well as giving ourselves feedback and what we could’ve done differently now that other teams have also released their solutions. 

Rankings breakdown by round

| Round | Ranking              |
| ----- | -------------------- |
| 1     | 668                  |
| 2     | 274                  |
| 3     | 23 (Our final round) |
| 4     | 259                  |
| 5     | 804                  |

# IMC Prosperity

IMC Prosperity is a global trading challenge hosted by IMC Trading that challenges STEM students worldwide to a 15 day trading competition. They aim to simulate real world markets and require the participants to apply Python programming skills, strategic thinking, and analytical abilities to the context of algorithmic trading virtual assets. 

This competition is separated into 5 rounds plus a tutorial round and includes both an algorithmic trading and manual trading section, with the latter more similar to a set of puzzles that one needs to solve to gain a small edge over other competing teams. In the end, the manual section did not make up a significant amount to the top team’s PnL, but was a fun set of exercises to complete for a break. 

More information about the competition can be found here, as well as this year’s wiki going into more detail about the specific attributes of each round. 

# Round 1

*Algorithmic*
In this section, IMC introduced rainforest resin, kelp and squid ink. We had made a simple market making strategy for rainforest resin with hard coded values in the tutorial stage assuming that the true value does not move/fluctuate from a fixed value (10,000). When backtesting using the website data, we find that our algorithm was not performing as well comparatively and we turn to optimise our methods. Improvements included simply increasing the number of orders as we weren’t hitting our position limit, increasing our buy/sell spread based on the mid price, and the inclusion of passive quotes. This method regularly hit the position limit and we had to include methods to scale our orders based on our position. There were talks and methods of resetting one’s position by using 0 EV orders however we didn’t fully understand how IMC handles orders hence did not implement this feature. 

For kelp, we started off using the same strategy as rainforest resin and used a moving mean as the fair market value. This strategy did not work well and we quickly gave up on it. Now that the competition is over, it’s regrettable to see that all we needed to do was reduce our spread as after normalisation it was near identical to rainforest resin. We pivoted to an autoregressive (ARX) model that predicts the next value in a time series using a linear combination of previous values within the same series. We use the past 5 mid prices in combination with the spread and the imbalance of the order book as input variables to determine the expected price. We found our optimal coefficients to be more heavily weighted towards the more recent prices, as well as the spread and imbalance having a negative influence, regularising “overreactions”.

Squid ink was a more volatile product compared to rainforest resin and kelp. It has a relatively tight spread with price jumps seemingly randomly. The IMC generously informed us that this product was mean reverting in the short term, and a metric to keep track of the size deviation from the recent average would be helpful. In response, we used the last 20 price z-score as an indicator for our trade entry logic, and closed our positions appropriately when the z-score reverts to 0. 

For all three strategies, we tried to market make wherever possible in situations where there were inconsistencies or where profitable. Passive quotes were only placed for rainforest resin as we didn’t know if it would be beneficial for kelp or squid ink. In total, round 1 strategies generated 42,000 seashells per round due to the consistency of market making. 

*Manual*
The manual section for this round was straight forward, finding the optimal combination using a brute force approach netted us approx 9% profit.

# Round 2

*Algorithmic*
Round 2 introduced to us three more products, as well as two baskets. Three products included croissants, jams and djembes, and the two baskets contain a set amount of each of these products (basket 1: 6 croissants, 3 jams, 1 djembes, basket 2: 4 croissants, 2 jams). It was also an option to trade these products individually, each with their own new position limits.

Our strategy was to trade picnic baskets when its market price deviates from its synthetic fair value, and exploiting inconsistencies. For the picnic baskets, we compare the fair value to the mid-price at the current timestamp and use a sigmoid-scaled factor to size orders proportionally to this difference. We also tracked the hedge volume to neutralise the risk using croissants, jams and djembes. This is to offset the exposure created from our basket trades. Offsetting our trades using hedges is a tradeoff we were happy with, reducing the variance but also reducing the expected PnL. 

*Manual*
The manual section of round 2 a game theory and probability question. Shipping containers get washed ashore with valuables (10k * multiplier) inside. Each container has a multiplier label and a inhabitants (up to 10) label. You are able to choose one container for free, but the second will come at a cost of 50,000 seashells. The profit is calculated by dividing the valuables by the sum of the inhabitants and percentage of the participants that choose it.

We looked at the question from a game theory only perspective and avoided the supposed Nash equilibrium, as the choices of other players did in fact have an effect on the outcome. We ultimately ended up with 31x, although I’m unsure how that was decided as I didn’t partake in helping the manual section for this round. 

# Round 3

*Algorithmic*
In round 3 introduced a derivative instrument called volcanic rock vouchers, which gives you the right but not the obligation to buy volcanic rock at a certain price (strike price) at the expiry timestamp, similar to a European option. There were different types of vouchers for different strike prices from 9500, 9750, 10000, 10250, 10500 with the underlying asset – volcanic rock –  priced around 10,000. 

In the hints section, it was explained that we should plot v_t and m_t where v_t = BlackScholes ImpliedVol(St, Vt, K, TTE) and m_t = log(K/St)/ sqrt(TTE) where t: Timestamp, St: Voucher Underlying Price at t, K: Strike, TTE: Remaining Time till expiry at t, Vt: Voucher price of strike K at t. We did have an algorithm that used the Black-Scholes model for options pricing, however our delta hedging strategy worked well enough and around this time was when our team started to get busy. 

Our strategy focused on pricing and arbitraging the volcanic voucher by building a dynamic options model. The model price is approximated as the sum of the intrinsic value and the volatility premium over a moving window of past prices. We included risk management through delta hedging, which was a proxy for the call option’s delta, representing the sensitivity of the option’s price caused by the underlying derivative. This allowed us to capture outstanding movements that would otherwise have been detrimental to our PnL. By estimating the delta, we could also calculate an appropriate hedge volume in volcanic rock. This allowed us to neutralise our risk similar to the hedging done in round 2’s algorithmic section by reducing the variance of our PnL and realise the arbitrage profits while being more neutral. 

Our success in round 3 shot us all the way up to 23rd place (after manual section correction), with a gain of 283,000 seashells through the algorithm section alone. 

*Manual*
The manual section of round 3 is setting the right price to buy flippers off sea turtles, which can then be sold later on to realise profits for this manual section at 320 seashells each. Each sea turtle will accept the lowest bid that is over their reserve price. The distribution of the reserve prices is a piecewise uniform distribution from 160-200 seashells and from 250-320 seashells, 0 everywhere else. You get to bid once in this manner, and the second bid will be similar, however the reserve price will be the average of all second bids instead. But unlike the first bid, your second bid will still have the chance to be traded, however that probability decreases rapidly the further down you go from the average. 

The optimal and expected gain was calculated to be at p = 200, however I had doubts in my own code and went with my teammate’s solution of 195. This resulted in an 8% loss compared to the optimal for bid 1. For our second bid, the team went ahead with 295 (optimistic bid above assumed average), 10 more than 285, which seemed to have been the consensus within the discord for the optimal second bid. With the final average being 287 (I assume with people trying to +1 the average, and those who are +1’ing those who bid 286), we sat comfortably with 50,000 seashells, 4,000 seashells/8% from the optimal. 

# Round 4

From this point on, there will be no further comments on the algorithmic section as we did not code anything. However, I did give the manual section of this round a go before I had to head off. 

*Manual*
The manual section of round 4 is nearly identical to round 2. We are also given access to the results of round 2 such that we could be more informed as to our decisions. Much like in round 2, the number of contestants can be thought of as sensitivity in the sense that the more contestants there are for a given suitcase, the less affected you’ll be by an additional person selecting that suitcase. Now that the round 2 results are out, we can see that the combination of 10x 20x being the most risky was also the most sensitive to change. 

This round specifically we were given access to opening a third box for 100,000 seashells. I was under the assumption that many people would choose a combination of the top few suitcases that had a high “scaled” multiplier, and given the choice of a second suitcase being available diluting the “percentage of players that choose this suitcase” variable, choosing a suitcase with a lower contestant count was not completely out of the question. I ended up choosing 80x and 50x, but swapping the 50 to 60 last second due to greed. Both had scaled multipliers of over 7 and both not the most attractive for other players, resulting in we had a final profit of +80k seashells. By pure chance, 80x was one of the more optimal choices, with the most optimal combination being 79x 83x. I don’t know how to predict human greed. 

We jumped from rank 800 to around 140 for the manual section after round 4.

# Round 5

We did not attempt round 5, and had left the competition completely by this point. 

# Ending statements 

If you made it this far, thanks for reading. This trading challenge made me truly realise how prepared you need to have been when an opportunity presents itself. Many of the hours I spent cluelessly looking at graphs and numbers after throwing random data analysis libraries and AI prompts, reminding myself that there’s people out there that know what to look for is an incredibly exciting feeling. While at times demotivating, it’s good knowing the only way you can progress is forwards when you’re a beginner at the starting line.**