## [Gender Bias in Coreference Resolution:Evaluation and Debiasing Methods](NAACL2018_CorefBias.pdf) ##

[Jieyu Zhao](http://jyzhao.net/), [Tianlu Wang](http://www.cs.virginia.edu/~tw8cb/), 
[Mark Yatskar](https://homes.cs.washington.edu/~my89/), [Vicente Ordonez](http://www.cs.virginia.edu/~vicente/), 
[Kai-Wei Chang](http://www.cs.virginia.edu/~kc2wc/). NAACL-2018 short paper.

We analyze different resolution systems to understand the gender bias issues lying in such systems. Providing the same sentence to the system but only changing the gender of the pronoun in the sentence, the performance of the systems varies. To demonstrate the gender bias issue, we created a [WinoBias](WinoBias/wino/data/) dataset. 

The dataset is generated by the five authors of this paper. We use the professions from the [Labor Force Statistics](https://www.bls.gov/cps/cpsaat11.htm) which show gender stereotypes:
<table>
    <tr>
        <th colspan="4">Professions and their percentages of women</th>
    </tr>
    <tr>
        <th colspan="2">Male biased</th>
        <th colspan="2">Female biased</th>
    </tr>
    <tr>
        <td>driver</td>
        <td>6</td>
        <td>attendant</td>
        <td>76</td>
    </tr>
       <tr>
        <td>supervisor</td>
        <td>44</td>
        <td>cashier</td>
        <td>73</td>
    </tr>
       <tr>
        <td>janitor</td>
        <td>34</td>
        <td>teacher</td>
        <td>78</td>
    </tr>
       <tr>
        <td>cook</td>
        <td>38</td>
        <td>nurse</td>
        <td>90</td>
    </tr>
       <tr>
        <td>mover</td>
        <td>18</td>
        <td>assistant</td>
        <td>85</td>
    </tr>
       <tr>
        <td>laborer</td>
        <td>3.5</td>
        <td>secretary</td>
        <td>95</td>
    </tr>
       <tr>
        <td>constructor</td>
        <td>3.5</td>
        <td>auditor</td>
        <td>61</td>
    </tr>
       <tr>
        <td>chief</td>
        <td>27</td>
        <td>cleaner</td>
        <td>89</td>
    </tr>
       <tr>
        <td>developer</td>
        <td>20</td>
        <td>receptionist</td>
        <td>90</td>
    </tr>
       <tr>
        <td>carpenter</td>
        <td>2.1</td>
        <td>clerk</td>
        <td>72</td>
    </tr>
       <tr>
        <td>manager</td>
        <td>43</td>
        <td>counselors</td>
        <td>73</td>
    </tr>
       <tr>
        <td>driver</td>
        <td>6</td>
        <td>attendant</td>
        <td>76</td>
    </tr>
       <tr>
        <td>lawyer</td>
        <td>35</td>
        <td>designer</td>
        <td>54</td>
    </tr>
       <tr>
        <td>farmer</td>
        <td>22</td>
        <td>hairdressers</td>
        <td>92</td>
    </tr>
       <tr>
        <td>driver</td>
        <td>6</td>
        <td>attendant</td>
        <td>76</td>
    </tr>
       <tr>
        <td>driver</td>
        <td>6</td>
        <td>attendant</td>
        <td>76</td>
    </tr>
       <tr>
        <td>salesperson</td>
        <td>48</td>
        <td>writer</td>
        <td>63</td>
    </tr>
       <tr>
        <td>physician</td>
        <td>38</td>
        <td>housekeeper</td>
        <td>89</td>
    </tr>
       <tr>
        <td>guard</td>
        <td>22</td>
        <td>baker</td>
        <td>65</td>
    </tr>
       <tr>
        <td>analyst</td>
        <td>41</td>
        <td>accountant</td>
        <td>61</td>
    </tr>
    </tr>
       <tr>
        <td>mechanician</td>
        <td>4</td>
        <td>editor</td>
        <td>52</td>
    </tr>
    </tr>
       <tr>
        <td>sheriff</td>
        <td>14</td>
        <td>librarian</td>
        <td>84</td>
    </tr>
    </tr>
       <tr>
        <td>CEO</td>
        <td>39</td>
        <td>sewer</td>
        <td>80</td>
    </tr>
</table>
(Note: to reduce the ambigous of words, we made some modification of these professions: mechanician --> mechanic, sewer --> tailor, constructor --> construction worker, counselors --> counselor, designers --> designer, hairdressers --> hairdresser)