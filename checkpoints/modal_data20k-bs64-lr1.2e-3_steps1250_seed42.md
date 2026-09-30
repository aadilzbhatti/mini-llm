# Sample report

- checkpoint: checkpoints/ckpt_blk128_emb256_head4_layer4_bs64_steps1250_lr0.0012_minlr2e-06_seed42.pt
- step: 1250
- params: 16,105,297
- config: {'vocab_size': 50257, 'block_size': 128, 'n_embd': 256, 'n_head': 4, 'n_layer': 4, 'dropout': 0.0}
- eval_train_loss: 5.936040925979614
- eval_val_loss: 5.930332767963409
- full_val_loss: 5.959063722443843
- max_new_tokens: 256
- seed: 1234
- block_size: 128
- device: cuda:0

Generation stops when EOS (50256) is sampled: EOS is a document boundary, so text past it would be the model starting a new document.

Context note: the window holds 128 tokens, so with 256 new tokens every prompt has left the window by generated token 128; everything after that continues the model's own output only.

## Sampled, temperature 0.8, top-k 50

### definition

prompt: 'Photosynthesis is a process that'  [6 tokens, starts scrolling out at generated token 123, fully gone by 128]

draw 1:

```
Photosynthesis is a process that is the more challenging to the key of your daily health.
The data of the problem is on this step is at the child.
The best. The following of the new approach and make sure to take a healthy for your dog (in a body.
A, the water is a good tool to your child’s time. In the amount of the product, it is the most appropriate. You will be helpful in least a problem, such you want to take the child to avoid the health and you to make you.
What your text is just a good person that is, when it will be a person's that the best. The course of the internet is at least an individual.
The student is the answer, the most type of the two different ways, then the more common.
My children have the problem on a bit of the best things, and the one is the first time to take the right to help in our time.
What is many tips to help to
A, we can also want to have the same year.
The goal are one day.
The most of the children are a bit of the main ways that also a good, but they are a variety of the best time, you are going at the time
```
[256 tokens, no EOS]

draw 2:

```
Photosynthesis is a process that makes those who are a time in the following. This is often very easy to make more effective for a strong range of the time.
- The key- The course, it does not the time to be good for people and the most important way there is the number of your body, which is some of the same work, the same or a child, and are that we are too a new in the best hand.
-
- How to your children that of each example?
- The most step, this can help work you have to have a good approach.
In conclusion, you can have been a week in the story of your students.
- How you need to be a list of the way of the children or then it is important to work on your body.
```
[stopped at EOS after 159 of 256 tokens -- the model ended the document]

### biography

prompt: 'Albert Einstein was a German-born theoretical physicist who'  [10 tokens, starts scrolling out at generated token 119, fully gone by 128]

draw 1:

```
Albert Einstein was a German-born theoretical physicist who led on the island, which was the two years to the first was not called the same number of the war.
In the early-called name, the world was built a large war-up from the New York and its first of the French century, and the government were the the war, with the year, as the war of Jesus, which the church it was not only the first, of “no” are in this case of the world. I has to the first and the two years. When that was no first of the island, there is the years of a person, which was not been a lot of the history of the world that is only a single-the largest one of the population that to take a number of the one year to the city of the same region.
The "The highest the U.S.S.e.S.S.g. The first one period of two of the United States, the first of the world and in the New York was known to be the great place.
- The study of the country and the “The world’. This is more than one one of the new time for the people.
“In the original, in the history” the “
```
[256 tokens, no EOS]

draw 2:

```
Albert Einstein was a German-born theoretical physicist who also have an unique population of the C. It is more useful to the same group of the most of the United States. The only the population and the same role between the region. One of the second was that the city is very common. However, the world has been a particular part of it is not not that he is not,’s first very more likely to be too one and in least.
The most the most common reason of man was the most likely to be a bit of the “c” or his a great way to the state to be a few years of the first, which is going to see that, but the reason of a long-m would also be an effective sense of human.
There are this is the most two other time that all of the way it is important to be able.
It is the most important choice of the “The most common part of the first way to be important to be part of the amount of my new side of the other years.
“The first is one possible to do not have, what’re all of two years and the children” she said? As the new people with the children.
In the work, there is that “the day
```
[256 tokens, no EOS]

### science_explainer

prompt: 'Oxygen is a chemical element with'  [8 tokens, starts scrolling out at generated token 121, fully gone by 128]

draw 1:

```
Oxygen is a chemical element with the type of the temperature of the body and the cells for the body which is more effective.
```
[stopped at EOS after 19 of 256 tokens -- the model ended the document]

draw 2:

```
Oxygen is a chemical element with a higher condition.
Mums in the use of this type of the blood-term disease (b., and al.S., 3, 13 +10,000 et al.5,000-2 or the most effective of the body and any other components between the risk of the symptoms.
S. Z. b.g. P., in the study, B.e. D. J.S. A. (3.13.8).
8.1.
1; 2.12. P. B., E. J. B. et al. (d.S.2.1.org/01.10.doi., the E.10.3.01, 0. 0.00.3. 2.4-24.4.
- 3. (422-16.2.
- T.8. (2)
- 5.
- 3, 2. (1.3.gov.9.org. (1.5.) (1.0-5/2).
- The United.5-2.6/0 0.8.62.6.5.2.1.2/1,100)
- The United States
```
[256 tokens, no EOS]

### instructional

prompt: 'In this lesson, students will learn how to'  [9 tokens, starts scrolling out at generated token 120, fully gone by 128]

draw 1:

```
In this lesson, students will learn how to use their communities.
How should write your personal school. When you are a good school is a sense for one time.
You’m your child is one of a student’s a child’s a few years. These in my own language is a good source of a product, but you can see that you are likely to find.
In this essay is the word as a more important way.
How does I need to use the information about, you have?
What are the answer is the best thing you want to do you need?
There is a�s to ask. For example, the essay the following a good essay in any day, you need to use people about how much?
"If you can want me at the best book, you're no different about what you have.
S.S
My, I’s not the best way to be able to read the best.
The first year in the American Science
In June. The
What can’t the following
How’t get an overview of the “There will be a good person?” The first person’s a lot of what we want to try to the fact about how to that the
```
[256 tokens, no EOS]

draw 2:

```
In this lesson, students will learn how to be a part of the state.
A page is a good-being that will be the best or if you might be sure to you’re too one of their children.
- The word! If you’ll see what there is a few years of a lot of a school, you can use your home.
So you’re made you, I’s what’ll do you want to see what you have a week to see. Make your time I’ll have to use as you may have to see how them with the right. You can see you need to know about your dog?
The essay makes you a lot of your child.
What is your child really be a dog with your body?
There is a good look on a look at the health, you can make sure you!
We want to ask your right to your skin and have a person of the children to make you should make what to make.
I are a lot that is a good way to do you a doctor’t get sure and be sure that you can need to try to them that you do you have to help understand for the way it if you are the teeth. If you have your doctor who are that
```
[256 tokens, no EOS]

### bullet_list

prompt: 'There are several benefits to regular exercise:\n- '  [11 tokens, starts scrolling out at generated token 118, fully gone by 128]

draw 1:

```
There are several benefits to regular exercise:
- n.
- C.g.- S. A. (1/3.2), is the major effect of a high-known solution.
-
1.e.S.S.m. This has a key thing in order to make the treatment of the risk of the time with and a more difficult for your body.
4.3.6.S. You can be done by a lot of water and with a way.
- The time.
It is a way that is not a person’s best.
2. What is the use of this is one. There
The
"What we want to take the a�s?
For the child’s you is the most possible-term, then you’ve have no.
In the "p. and if you see what you know you do you don’t know if you will find some of a dog’t. If you are what you are any important to do not the best kind of the way of our hands.
```
[stopped at EOS after 217 of 256 tokens -- the model ended the document]

draw 2:

```
There are several benefits to regular exercise:
- . A: The most common time is a comprehensive way to the user and it is not the same time.
- You do the basic condition, you’s your pet, and you have a good way to the best cause it, and a lot of you can be a number.
- The person can also know this a body is so about a small time.
- Have the main reasons that you’t be very important – you’re just in things.
- To get the best way to know it and then many different ways if it’re really.
- It’s not one to get more useful to eat an important option, such as well as it you do not feel in time?
- Make sure you can have a common effect from your child and you can make to get a day.
- For instance in the number, you can know the answer of your body is the dog that it is the body to be a good food and you can be sure to know your doctor when you.
- Learn and your cat to make the teeth for a doctor which are much more to be more important. It can get sure much a doctor to make your your skin too best.
- You get a
```
[256 tokens, no EOS]

### numbered_list

prompt: 'To solve a quadratic equation, follow these steps:\n1.'  [15 tokens, starts scrolling out at generated token 114, fully gone by 128]

draw 1:

```
To solve a quadratic equation, follow these steps:
1.3-year-shaped and more than a healthy level of the same type of the process. From the case is the most effective and the risk to the test was used to the data and energy. If the type of the water is not been an longer better tool in the system is to be used by a long-quality and a more difficult approach.
- The case is the next different aspects of the risk of the first-specific activity, a few categories of the case.
- The project of the following one year,000 percent of the process, of the case and the system and the same time that allows them to get an important source of a variety of the time.
- The ability to get a significant factor that is that is a lot of the problem of a person.
- What are the most important way, and a problem?
- A. The first type of the best time is all or no important to be a more common change for the type of.
However: Some children that
- What is a most common thing to the other people it should be able to know that, you are, or we will be the most common and even then a great problem and the time of your home.
- Avoid you may do not
```
[256 tokens, no EOS]

draw 2:

```
To solve a quadratic equation, follow these steps:
1. The most common idea of the immune and can be done with the way to keep it. (P) The problem of the body is used at the potential of this and type on the brain, which was the most frequently known as it is in which the the same-free.
2. It is a single-tation, a high-being of the virus-the body is one.
2.
8.S. I’s the majority of those who was a few days.’s most common.
How have the best of this?
B.com. What is the most one of my mind is also many ways that in different types of the problem?
What is an most important to help a range of the most of any common time?
The time is the next day, and it is the second day is not different, a series of the most thing for the book is the end of the time.
’s a piece of a one, we will be more likely to avoid the same time.
How does I have a�m?
In the first one of the most one or a day?
He did you're it is, not many people to make the right to do to be able
```
[256 tokens, no EOS]

### enumeration

prompt: 'There are three main types of'  [6 tokens, starts scrolling out at generated token 123, fully gone by 128]

draw 1:

```
There are three main types of the second year of the United States to the United States, the Norths,000 million year of the 18 and 0.12% of the United States at the 17th century, which has been considered no common difference that the average-year-year-level.
1.5.6.12.2, 3:1.0.4,0.
3(2), 0.5.D. 3% 1 (1, 2.60 (1) in the U, with a state. This method would be the first used; to be a-old-free of the U.2. (Fig.15-19-50) 1
4:5.2/3. 1.1/S.C.8 2.3:2
3.
|10/2.C., 3.12.3
3.
5.5
|2 (30/5)
- H.
- 9.
- 2. 6.org.
- M.
- 10. 5. C. J.
- "2015)
```
[stopped at EOS after 226 of 256 tokens -- the model ended the document]

draw 2:

```
There are three main types of the number of the region and the number of which was the researchers the main.
The findings with the time of the most common evidence of the population is, which the number of the first in the world.
The second month that the first was taken with the other of the number of the study.
The last year of the research and the most likely to be used in the year and the world of law, to be known in the most of the World War. The National Journal of the West (F) would be the most important to the United States. The North Zealand is the ‘e.’
To note that I think that, it can be a number.’s only a way to change on the most side of the city.’s very well as part of the Uniteds of all of the last of the largest-being, with the same time, and the last the same amount of the work as a second time of the first. But if we will take a particular difference if “not the most to have a few to "in, the last few States"’s history, which is being a few days of the story.
The story of the world will take some of the idea that will be
```
[256 tokens, no EOS]

### long_dependency

prompt: 'Although the treaty was signed in 1919, it'  [9 tokens, starts scrolling out at generated token 120, fully gone by 128]

draw 1:

```
Although the treaty was signed in 1919, it found on India, but to be considered the first of the British War, the United States, the United States's “The law of Israel”. The main was the history of the United States, the American was a few of the largest, which were not a most of the American African population. The largest, and the national, the country, the world was a few decades of the American, the last, but he was not not the two of Europe, “the Worldism” to be used up to the ‘“The government, the other the people of the world is, and the second world in the world.
The country was no longer for the population.’s one of the first person, the world’s, and is two years of this section. The United States of a country of the city’s ‘” (e.S.’s world, the first man, the island, which ‘L.’’s a “a”, ‘a” and’s history was the “s a way to show that the people”. The first person is to find of a very small age of the city.�
```
[256 tokens, no EOS]

draw 2:

```
Although the treaty was signed in 1919, it was that a one thing of the "of to be on the first, the first "ch of the United States. Then that his wife the New York’s “s”'s death of he created by the first time, and.
The first is just the “I didn’t have the best hand when a child, a place, as we might know,”
’s very very powerful in the first year,’t really’s not an example in the world’s people. Even we’m was said. He said one of you’t be more less common for a year-being of their own work, you have a good role of a sense of the first book!
To look out the history, that the first-day life was being used for the most small time in order to be able much as to what you can eat the right by and in the child and the students will need for your ability.
There will be a few common number of two years.
Why it is a part, that that is the most one in a couple of people. But. A “An important that you can not you see how it is possible.”

```
[256 tokens, no EOS]

### agreement_gap

prompt: 'The students who had spent the entire semester preparing for the final examination in organic chemistry'  [16 tokens, starts scrolling out at generated token 113, fully gone by 128]

draw 1:

```
The students who had spent the entire semester preparing for the final examination in organic chemistry and other parts of the federal Ocean. For the next of the United States, then the number of the most than the United States. He is that it will not be a good role in the study of government on the second day, including the two years ago, but the other of his state and the most the United States.
How we have to do the study of each month, but the “We” or a more easily understand on the American the most two States” for the last year, I had a long-called example of this day.
- 5. What is the country, and the United States is the United States and the United States is the U.g.S.e.S. The following, however, we is just the most important for the world, or the United States.
The city was the most important part of the United States, but the end of his government would not not be a great part of the one side time. (p., where, if the study of the United States was born, and he is the great of the last government as to be the “a,” and the United States, “b-S.,”.“An York of a nation
```
[256 tokens, no EOS]

draw 2:

```
The students who had spent the entire semester preparing for the final examination in organic chemistry. The purpose of the city-f-day study-based people were able to provide their and support of them in our lives. We are at the last year, we use the world, the United States, the U. We have an interesting example. It is a part of education and the same time to create a number of a state as a very single number of schools, and all other countries of the U.S.S.S.S.
How does you look? They will be a day?
I have a�s, you should know how to get sure we go at the way to take, we use from the case where the school has to be done by a school. There is other ideas after how to get sure you see the need of one's.
The course of the best one section for the idea of the world and is a number of the course where the point is very useful.
The new day and we are about the most time is a little time when the first can be more in the story and this is, or we will see an argument of the book to be a great way to change and is the most important.
In order, this essay is a person. By to see, the following hand.
```
[256 tokens, no EOS]

### attribution

prompt: 'According to a study published in'  [6 tokens, starts scrolling out at generated token 123, fully gone by 128]

draw 1:

```
According to a study published in India, and a group, but the world may be used by the U and the world of the University of the number of the time of the two of the United States, and the largest, it is also a group of the end of the number of the world and the world.
You is you all to use the number of a world of the world, but we think that the world is not only about the world.
In the United States The two people and his year, one was the great reason that I did a group. From the first year, the following of the U.S.S.S.
It is not one of the largest American year. He is the "s to the time of the first of all a result of the world that the first is of the same kind of the city was being to do the main species of the American War, the man’s first year,’s death, which the “” one part of the New York century, was a few people that the world being in the two main-old. The country was in the most common history of the same-making, and the land.
The first-day the most common world is that this is the first much common.
```
[256 tokens, no EOS]

draw 2:

```
According to a study published in the state of the world is the most important to the most part of the world and the following a “mal-day,”, the most time is based in the U.S.C.S. S.S.e.S.S.S/1-S. 2; 7-1/e.org.7.0.org. doi.
1. 10-year-s, 2019.
- 1.
- The first is at a few months of all who can be the most important to be an part of the same parts of the following hand, the person is a child.
- There are the first one of the second.
- We are the one of the most two years, you are not a high?
- The first way of what is the way of the time? You can have the reason that the number of the second is the day.
- The idea of the children are the first is the best who are no important to do the “or”, and the term that does not be the work in the people.
- “The child’s one of the people who are the way.”
- The first is what I be aware
```
[256 tokens, no EOS]

### dialogue

prompt: '"I do not think that is correct," she said, "because'  [14 tokens, starts scrolling out at generated token 115, fully gone by 128]

draw 1:

```
"I do not think that is correct," she said, "because for the past of the people and his own one of the first. He is the first one that is not the “b” I had to know me, which I’m. I believe that I did not had a time of how the other things is the most of our work and then the only was the people, and it’s an sense of this case of a part of the right.
’s some people who’s just just a real-day sense for the time, and they cannot be more than that would become the number of.
- What we learn you do you just the I have do you do we are that this was no important to be.
At the first way that that you have a very able to be an common number of a number of the school, is still a number of the school, but you are this is an important. The reason may say that the world is not one of the same other people who are to see a lot of its own life for the body.
- The same thing to make a lot of our own life, which is so this means you to think that the time is a more important sense,, of which it will be a positive.
The concept
```
[256 tokens, no EOS]

draw 2:

```
"I do not think that is correct," she said, "because to me, who were there’s a way to see this will be a sense of the other and in the “as’ to do, but they may see which he had it in this, I might have a problem?” The first important with this. He made that I would also not the right and I don who don’t think to be of the question, and I’t just, but I didn.
How do the most, a “in to say it's the word.”
If ‘There a man’. “�the other person is a good way to look to find, or then that the time,” said I need to do that his right for the way as we get a few weeks of the time so.”
However, we have too part of the most of the time of the fact.
”
How is a I never really know what does not work when we want to do I’t have only a long-as that is a time.
What do it’s the case of this is to do.
’s the name of this is a new?
The other most important way of the word
```
[256 tokens, no EOS]

### factual

prompt: 'The capital of France is'  [5 tokens, starts scrolling out at generated token 124, fully gone by 128]

draw 1:

```
The capital of France is a new state that is a total of the American Africans.
The main study of the Cills were not no common with the first common way to the “a” but the city of the United States’s the United States, the European Empire had been the most more famous and their family or the state to be created. This is the researchers found the first-19% of two main events, an opportunity to which are still to be seen at the government, and the same part of the country in the U.S. (C. (4.org: 3,000).
S.e.S. In this, we take a higher number of the U. and the second year.S. I.g.S.S.
The country had a result of the life of the second year. The second person in the year had to show a significant time of it is not, and the same study is the entire year, the world’s or the “What is any key for. This is that the way that the year is that they are an second in these people.”
My first of what the day is the second time is a good that is not not to be much common than the
```
[256 tokens, no EOS]

draw 2:

```
The capital of France is the largest the Uth century in the United States. The South Atlantic of the U. The U. 18th century and the state were the main part of the late War, which a single group of the South Africa. The government was a group of the highest, and his population and the "b, the city’s), but it was some to the first time of the two. In the United States, the first of the government of the British and most two-century society that was given.
“The world “The U.”.S.S. (C.e.e/c.n.S. (A.D.).
T. “H.”, with the United Nations. ‘B.’ “In the “F.”
S. “The “I would say when I have only think, I will use, and not on this.
What is my “You”.’s about the best time, and that we are even if you is the great to see a day the time and all?
’s and I do you’re your look a good question?
For sure that I�
```
[256 tokens, no EOS]

### numeric_units

prompt: 'The mountain rises to a height of'  [7 tokens, starts scrolling out at generated token 122, fully gone by 128]

draw 1:

```
The mountain rises to a height of the southern culture.
The latter is also been an large in the United States. It is also made that he comes from its first, but in the last year.
It was one was two different who was the most significant and first of the last country to the government of Europe. The first was born before the first few days of the same case of the United Nations and the country, and the end this.
The National War also held in a group-s. He said, was found in the time of the United States. The study was the U. "The main number of the two days, we also are done by to each year, and how would be one, we are in the way.
In the state of the new one.
The most time of the United States we have taken in the history, then a number of the News, the number of the future and the country are, there is the main the most common as a high-specific number of the world.
After both about the United States and the case of the population was the total of the first group. The region is the same common group that is also not the best.
The first time that the government’s one is an important part of the
```
[256 tokens, no EOS]

draw 2:

```
The mountain rises to a height of the western in the American population. The study of the UK by The World War is for the United States.
There are the American and the American study and the city and the population.
As a recent period of the United States, the highest the global population was the most two. The number of the U.S.S.S.S.S. (4.,200
1.2, and 2.
5. 5.1/3 at the population, 3.3.
0.5.3.6.4.30 (10). doi:5.212.4.3.1||1.2

2
2, N.
3. In 10% and 2.6. It is in the C.g.S.1.
2) B.4/10
2. The D: 0.1. 2
6.30.7. A.7 2 of the (9/1. 2.1.5.6.6.3.9.g.5.9.
- 5.7.5.3.2.2/2.8
D. [8.3||2.2.7 5.1.2
```
[256 tokens, no EOS]

### code_ood

prompt: 'def fibonacci(n):'  [7 tokens, starts scrolling out at generated token 122, fully gone by 128]

draw 1:

```
def fibonacci(n): A.pdf/tor-f. ISBN.com-A., & et al.g.g.
- A. A.
- A. W.
- 3. C.
- 2.S. S.
- A. D. (1.1.g., pp. (1. 5.com.g.1): doi.
- The C.S. (0.8. 5, 2014).
- 1.S/2. doi. (20):
- D. D.e., p.S.. and the U.g.4. (0.2–1.e.com).
- G. J
- G.e. L. (10.S Scholar), K. (2016/S.
- H.
- C. "2007), 5. doi.13.10.1. (10.1-3, 2.org/2/2007.g.com-g. [1. 1-10/3.4.gov.5.com.
- I, C. R. K., 4, & P. (d. E. (2003).
- 4. 3.
- 5/
```
[256 tokens, no EOS]

draw 2:

```
def fibonacci(n):
- “L. (T): M. et al. (m.)
-g. L., S.
-
-, there are the most important to the most common, that the only known as in the same time.
- The best of the following-g.
- The two types of a few other types of the population and the same other.
- The C.
- In this was the United States, or the number of the last day by the first-and, and the first.
- The "c.e. C, p., and it, the last of the same case of the “the first’s and the people, it is being the most difficult to get them to be no important or more longer. If you've use the person is the main days of these years of that is a lot in time. They are actually the right to be in which the other ways has no problem to be more than that.’s a particular child is another-t, and the time.
- So, I am the day that’t do, we’t have to see the first week.
-
- If the following year of the first the following words
```
[256 tokens, no EOS]
