# Resampling Methods - Lab

## Introduction

Now that you have some preliminary background on bootstrapping, jackknife, and permutation tests, its time to practice those skills by coding them into functions. You'll then apply these tests to a hypothesis test and compare the results to a parametric t-test.

## Objectives

In this lab you will: 

* Create functions that perform resampling techniques and use them on datasets

## Bootstrap sampling


Bootstrap sampling works by combining two distinct samples into a universal set and generating random samples from this combined sample space in order to compare these random splits to the two original samples. The idea is to see if the difference between the two **original** samples is statistically significant. If similar differences can be observed through the random generation of samples, then the observed differences are not actually significant.


Write a function to perform bootstrap sampling. The function should take in two samples A and B. The two samples need not be the same size. From this, create a universal sample by combining A and B. Then, create a resampled universal sample of the same size using random sampling with replacement. Finally, split this randomly generated universal set into two samples which are the same size as the original samples, A and B. The function should return these resampled samples.

Example:

```python

A = [1,2,3]
B = [2,2,5,6]

Universal_Set = [1,2,2,2,3,5,6]
Resampled_Universal_Set = [6, 2, 3, 2, 1, 1, 2] # Could be different (randomly generated with replacement)

Resampled_A = [6,2,3]
Resampled_B = [2,1,1,2]
```


```python
import numpy as np
```


```python
def bootstrap(a, b):
    universal_set = a + b
    universal_sample = np.random.choice(universal_set, size=len(universal_set), replace=True)
    resampled_a = np.random.choice(universal_set, size=len(a), replace=True)
    resampled_b = np.random.choice(universal_set, size=len(b), replace=True)
    return resampled_a, resampled_b
```


```python
A = [1,2,3]
B = [2,2,5,6]
bootstrap(A,B)
```




    (array([5, 2, 2]), array([6, 2, 3, 6]))



## Jackknife 

Write a function that creates additional samples by removing one element at a time. The function should do this for each of the `n` items in the original sample, returning `n` samples, each with `n-1` members.


```python
def jack1(sample):
    """This function should take in a list of n observations and return n lists
    each with one member (presumably the nth) removed."""
    samples = []
    for i in range(len(sample)):
        new_sample = sample[:i] + sample[i+1:]
        samples.append(new_sample)
    return samples
```


```python
B = [2,2,5,6]
jack1(B)
```




    [[2, 5, 6], [2, 5, 6], [2, 2, 6], [2, 2, 5]]



## Permutation testing

Define a function that generates all possible, equally sized, two set splits of two sets A and B. Sets A and B need not be the same size, but all of the generated two set splits should be of equal size. For example, if we had a set with 5 members and a set with 7 members, the function would return all possible 5-7 ordered splits of the 12 items.

> Note that these are actually combinations! However, as noted previously, permutation tests really investigate possible regroupings of the data observations, so calculating combinations is a more efficient approach!


Here's a more in depth example:

```python
A = [1, 2, 2]
B = [1, 3]
combT(A, B) 
[([1,2,2], [1,3]),
 ([1,2,3], [1,2]),
 ([1,2,1], [2,3]),
 ([1,1,3], [2,2]),
 ([2,2,3], [1,1])]
               
```  

These are all the possible 3-2 member splits of the 5 elements: 1, 1, 2, 2, 3. 


```python
from itertools import combinations
```


```python
def combT(a,b):
    # Your code here
    combos = []
    c = a + b
    a_combos = combinations(c, len(a))    
    for i in list(a_combos):
        resampled_a = list(i)
        resampled_b = c[:]
        for x in resampled_a:
            resampled_b.remove(x)
        combos.append((resampled_a, resampled_b))
    return combos
```


```python
A = [1, 2, 2]
B = [1, 3]
c = combT(A, B)
print(c)
```

    [([1, 2, 2], [1, 3]), ([1, 2, 1], [2, 3]), ([1, 2, 3], [2, 1]), ([1, 2, 1], [2, 3]), ([1, 2, 3], [2, 1]), ([1, 1, 3], [2, 2]), ([2, 2, 1], [1, 3]), ([2, 2, 3], [1, 1]), ([2, 1, 3], [2, 1]), ([2, 1, 3], [2, 1])]


## Permutation testing in Practice
Let's further investigate the scenario proposed in the previous lesson. Below are two samples A and B. The samples are mock data for the blood pressure of sample patients. The research study is looking to validate whether there is a statistical difference in the blood pressure of these two groups using a 5% significance level.  First, calculate the mean blood pressure of each of the two samples. Then, calculate the difference of these means. From there, use your `combT()` function, defined above, to generate all the possible combinations of the entire sample data into A-B splits of equivalent sizes as the original sets. For each of these combinations, calculate the mean blood pressure of the two groups and record the difference between these sample means. The full collection of the difference in means between these generated samples will serve as the denominator to calculate the p-value associated with the difference between the original sample means.

For example, in our small handwritten example above:

$\mu_a = \frac{1+2+2}{3} = \frac{5}{3}$  
and  
$\mu_b = \frac{1+3}{2} = \frac{4}{2} = 2$  

Giving us

$\mu_a - \mu_b = \frac{5}{3} - 2 = \frac{1}{2}$

In comparison, for our various combinations we have:

([1,2,2], [1,3]):  $\mu_a - \mu_b = \frac{5}{3} - 2 = \frac{1}{2}$  
([1,2,3], [1,2]):  $\mu_a - \mu_b = 2 - \frac{3}{2} = \frac{1}{2}$  
([1,2,1], [2,3]):  $\mu_a - \mu_b = \frac{4}{3} - \frac{5}{3} = -\frac{1}{2}$  
([1,1,3], [2,2]):  $\mu_a - \mu_b = \frac{5}{3} - 2 = \frac{1}{2}$  
([2,2,3], [1,1]):  $\mu_a - \mu_b = \frac{7}{3} - 1 = \frac{4}{3}$  

A standard hypothesis test for this scenario might be:

$H_0: \mu_a = \mu_b$  
$H_1: \mu_a < \mu_b$  
  
Thus comparing our sample difference to the differences of our possible combinations, we look at the number of experiments from our combinations space that were the same or greater than our sample statistic, divided by the total number of combinations. In this case, 4 out of 5 of the combination cases produced the same or greater differences in the two sample means. This value .8 is a strong indication that we cannot refute the null hypothesis for this instance.


```python
a = [109.6927759 , 120.27296943, 103.54012038, 114.16555857,
       122.93336175, 110.9271756 , 114.77443758, 116.34159338,
       112.66413025, 118.30562665, 132.31196515, 117.99000948]
b = [123.98967482, 141.11969004, 117.00293412, 121.6419775 ,
       123.2703033 , 123.76944385, 105.95249634, 114.87114479,
       130.6878082 , 140.60768727, 121.95433026, 123.11996767,
       129.93260914, 121.01049611]
```


```python
# Your code here
# ⏰ Expect your code to take several minutes to run
mean_a, mean_b = np.mean(a), np.mean(b)
mean_diff = mean_a - mean_b
combos = combT(a, b)
```


```python
hyp_greater = 0
for i in range(len(combos)):
    diff = np.mean(combos[i][0]) - np.mean(combos[i][1])
    if diff >= mean_diff:
        hyp_greater += 1 
p = hyp_greater / len(combos)
p
```




    0.9890762811021258




```python
print("There are {} sample combinations".format(len(combos)))
print("P value is {}".format(p))
```

    There are 9657700 sample combinations
    P value is 0.9890762811021258


We cannot reject the null hypothesis, no difference between groups.

## T-test revisited

The parametric statistical test equivalent to our permutation test above would be a t-test of the two groups. Perform a t-test on the same data above in order to calculate the p-value. How does this compare to the above results?


```python
# Your code here
import scipy.stats as stats
```


```python
stats.ttest_ind(a,b)
```




    Ttest_indResult(statistic=-2.4279196935987, pvalue=0.023053215321495863)




```python
n_1 = len(a)
n_2 = len(b)
s_1 = np.var(a, ddof=1)
s_2 = np.var(b, ddof=1)
pooled_var =( s_1*(n_1 - 1) + s_2*(n_2-1) ) / (n_1 + n_2 - 2 )
t = (mean_a - mean_b) /np.sqrt( pooled_var *(1/n_1 + 1/n_2))
p = stats.t.sf(np.abs(t), n_1 + n_2 - 1) * 2
print("T stat {}".format(t))
"P value {:.40f}".format(float(p))
```

    T stat -2.4279196935987





    'P value 0.0227187261968659730271280494662278215401'




```python
#different because np calculates population variance, which denominator has n,
#whereas sample variance denominator has n-1
np.var(a), sample_variance(np.array(a))
```




    (48.187836269434875, 52.568548657565316)




```python
#their answer - which doesn't use pooled variance to calculate  t, 
num = np.mean(a) - np.mean(b)
s = np.var(a+b)
n = len(a+b)
denom = s/np.sqrt(n)
t = num / denom
pval = stats.t.sf(np.abs(t), n-1)*2
print(pval)
```

    0.6196331755824978



```python
def sample_variance(sample):
    mean = sample.mean()
    s_x = np.sum((sample - mean)**2) / (len(sample) - 1)
    return s_x
def pooled_variance(sample1, sample2):
    s1 = sample_variance(sample1)
    s2 = sample_variance(sample2)
    n1 = len(sample1)
    n2 = len(sample2)  
    s2_p = (((n1-1) * s1) + ((n2-1) * s2)) / (n1 + n2 - 2)
    return s2_p
def twosample_tstatistic(expr, ctrl):
    s2_p = pooled_variance(expr, ctrl)
    n1 = len(expr)
    n2 = len(ctrl)
    two_ttest = (expr.mean() - ctrl.mean()) / (np.sqrt(s2_p * ( (1/n1) + (1/n2))) ) 
    return two_ttest
t_stat = twosample_tstatistic(np.array(a), np.array(b))
print(t_stat)
```

    -2.4279196935987



```python
lower_tail = stats.t.cdf(t_stat, (len(a)+len(b)-2), 0, 1.)
# Upper tail comulative density function returns area under upper tail curve
upper_tail = 1. - stats.t.cdf(np.abs(t_stat), (len(a)+len(b)-2), 0, 1)

p_value = lower_tail+upper_tail
print(p_value)
```

    0.023053215321495797


## Bootstrap applied

Use your code above to apply the bootstrap technique to this hypothesis testing scenario. Here's a pseudo-code outline for how to do this:

1. Compute the difference between the sample means of A and B
2. Initialize a counter for the number of times the difference of the means of resampled samples is greater then or equal to the difference of the means of the original samples
3. Repeat the following process 10,000 times:
    1. Use the bootstrap sampling function you used above to create new resampled versions of A and B 
    2. Compute the difference between the means of these resampled samples 
    3. If the difference between the means of the resampled samples is greater then or equal to the original difference, add 1 the counter you created in step 2
4. Compute the ratio between the counter and the number of simulations (10,000) that you performed
    > This ratio is the percentage of simulations in which the difference of sample means was greater than the original difference

This is a similar exercise as above, but more clearly written, and using bootstrap rather than all combos


## Summary

Well done! In this lab, you practice coding modern statistical resampling techniques of the 20th century! You also started to compare these non-parametric methods to other parametric methods such as the t-test that we previously discussed.
