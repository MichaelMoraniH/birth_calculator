The Purpose Of The Model:
------------------------
This calculator calculate the probability distribution of the effect (in terms of number of children) of having kids.
For an example, it can calculate the probability that if i have one kid, in 3 generations there will be 10 more people
in the world (compared to a reality in which I did not exist), or 5 more people, or 0 more (which means I will have
no effect) or -10 people (which means there will be 10 less people), etc, and the probability for effect>k, or effect>k,
or k1<effect<k2.

The calculation only consider the effects of having kids (for each person: me, my offspring, the person who would be
married to my wife/husband if I didn't exist, etc), and not their actives in their lives (killing and saving lives).

In general, the calculation based on "alternately marriage chain":
A is married to B, and B would be married to C in alternate reality (where A doesn't exist), and C is now married
to D... until X which is single now or in the alternate reality.


Two versions of the model:
-------------------------
The previous model was written in the form of an explicit calculation, in which the calculation - for a large radius -
took a long time (slow_version). Then we converted the code to code in tensors (fast_version), and thus the calculation became
significantly faster, but an accuracy problem arose. Read more about this in the explanation of Accuracy issue


The Assumptions We Made To Simplify The Calculation:
---------------------------------------------------
We make several additional assumptions designed to greatly simplify the model, so the results are less chaotic and
the calculation less complex.

We assume that only the people of A's gender determine the number of children, so if X is of A's gender then A affects
only his own children (from "not being" to "being") and X's children (from "being" to "not being"), but if X is of the
opposite gender - A affects only his own children.

We also assume that the changes of marriage within the chain do not affect the future of the heritage of any of the
children. For an example, we assume that the future heritage of the children of B+C in the alternate reality will be the
same as future heritage of the children of C+D in this reality.

In short, we just calculate, recursively, A_effects minus X_(possible)_effects, ignore any effect in the middle of
the chain or around the marriage.

Although these assumptions are unrealistic, we believe that the calculation still serves the purpose: the purpose is to
demonstrate the chaos caused by having children, in the sense that the probability of a zero effect or a small effect
is a very low probability, and that the effect can be very large, and also in the sense that the probability of reducing
the number of people in the world due to the birth of children is similar to the probability of increasing the number.
In other words: the effect of having children is expected to be very large, and very unpredictable. Our assumptions only
reduce the chaos in all respects, so in reality the chaos will be even greater.

Other simplifying assumptions are:
- The birth distribution and the probability of being single do not change over the generations.
- The tendency to a reduced or high birth rate does not pass from the parents to the children.
- The probability of each person being single in the alternate reality (even though they are now married) is the same
  as the average person's probability of being single.
- We omitted the option for a family with more than 4 children, and the option for a single-parent family. This is 
  because both of these options greatly complicate the model.


Calculate Approximations To Save time:
-------------------------------------
Unfortunately, the runtime complexity of the model is exponential. Therefore calculations for 8 or more generations
become impractical for the PC.

For example, after 10 generations there are more than a billion possibilities for effect size.

To deal with the problem, we truncate the range of effects down to the radius set using the RADIUS variable. During the
calculation, any value that exceeds RADIUS above or -RADIUS below is truncated. On the one hand, we don't want to
prevent the truncated values from continuing to participate in the rest of the calculation (after all, they can continue
to influence the small values), but on the other hand, we don't want their influence to be too high. That's why we
handle the truncated values in two different ways, which produce two different approximations:

1. We save the values higher than RADIUS as "more than {RADIUS}", and the values smaller than -RADIUS as "less than
   -{RADIUS}". They continue to participate in the calculation, where the result of adding "more than {RADIUS}" with any
   number (including itself) is still "more than {RADIUS}", and so is "less than -{RADIUS}"; Except for the case where
   these two special values are added. In this case the result of the connection does not participate in the
   calculation, and the probabilities are lost. Naturally, the approximation this method provides is a lower bound for
   the true values, throughout the range (it is always a lower bound for the values in the defined radius).

2. We add the values: RADIUS+1, -(RADIUS+1), and send them all the values that go beyond RADIUS and -RADIUS,
   respectively. These special values continue to participate in the calculation as usual, but in the end they are not
   returned - but only the values in the defined radius. Naturally, this approximation is an upper bound on the true
   values within the RADIUS range, so (since it preserves the sum of probabilities 1) it is lower bound for values
   beyond the radius (which are used to calculate ranges "to infinity").

We provide both approximations, and can say the true probabilities are somewhere between the two approximations, among
the effect values in the range.

The larger the RADIUS, the closer the two approximations will be to each other and therefore more accurate, but the
running time will be correspondingly longer. In general, we can say that our empirical tests revealed that estimate (2)
is much more accurate than estimate (1), except for the probability of 0 effect - in this case approximation (2) is
significantly much higher than the true probability, and approximation (1) is very close.

In addition, we calculate the sum of all the probability values that appear in approximation (2) (including those of the
special values), these are the values that were not lost, and present this sum as the precision.

Besides the truncation, the limitations of the computer also impair accuracy: the calculation performs many
multiplications of numbers whose absolute value is less than 1. The digital representation cuts digits after the point
beyond a certain index, and thus after many multiplications too low values are obtained. This deviation must also be
taken into account.


Accuracy issue:
--------------
Besides the bias created by the truncation, there is an inaccuracy that comes from the limited representation of numbers
on a computer. However, this inaccuracy only becomes significant for the lower bound in the calculation of the tensor
version. It also affected the accuracy of the upper bound there, but empirical testing showed that repeatedly correcting
the numbers (so that their sum is equal to 1, throughout the calculation process) restores the necessary accuracy.
Unfortunately, such a correction is not possible in the lower bound calculation.

For this reason, we also left the code of the slow version, where the calculation remains accurate anyway.
