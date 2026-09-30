# Sample report

- checkpoint: checkpoints/data10k_160k_emb256_tied_bias_seed42.pt
- step: 160000
- params: 16,088,913
- config: {'vocab_size': 50257, 'block_size': 64, 'n_embd': 256, 'n_head': 4, 'n_layer': 4, 'dropout': 0.0}
- max_new_tokens: 128
- seed: 1234
- block_size: 64
- device: cpu

Generation stops when EOS (50256) is sampled: EOS is a document boundary, so text past it would be the model starting a new document.

Context note: the window holds 64 tokens, so with 128 new tokens every prompt has left the window by generated token 64; everything after that continues the model's own output only.

## Sampled, temperature 0.8, top-k 50

### definition

prompt: 'Photosynthesis is a process that'  [6 tokens, starts scrolling out at generated token 59, fully gone by 64]

draw 1:

```
Photosynthesis is a process that a sense of the use of the same time, you can also get the impact of the different or a child.
It is being used in a few of them when to use: (n) and more effective) as an individual, and the year).
The study is a long one of a significant increase in the course of the people with a very easy for more common aspects of water health of our lives. We can help the two ways to go to use the same example:
- The most information is a lot of the problem of the time you can cause of your body and the same. If the system, it causes
```
[128 tokens, no EOS]

draw 2:

```
Photosynthesis is a process that a result in order to the right to be the use of the case of the following an appropriate and its future operations.
According to a sense of the case of the United States, the U.Ss, T, the US Medical Center.S. C. S, and the United States. (2010). The objective of the case of government is the largest, the country. This includes a result of the same period of the United States, it has been the United States, while developing the US. The first part of the last year, it is its own on the most important to the largest as a huge than 10.
```
[128 tokens, no EOS]

### biography

prompt: 'Albert Einstein was a German-born theoretical physicist who'  [10 tokens, starts scrolling out at generated token 55, fully gone by 64]

draw 1:

```
Albert Einstein was a German-born theoretical physicist who had ever been passed from 18th.
They were no longer allowed to be able to be found in the past 18, even the American Heritage of the “the most of the war of the British. This is the same time on its own way to do when the land’s going to be a person who is still more than the same time.
The people are not only one must see these birds like the end of the first thing. It is that we also a look for it's time, but they are some two times in a great manner.
“A good understanding is a few days.
So
```
[128 tokens, no EOS]

draw 2:

```
Albert Einstein was a German-born theoretical physicist who used a new in the United States. The federal government, the city and the ‘Sure, however, when there is that is not be found in the region into the country, including a result of the country. We began up their independence, although the nation, however, who did not even when its way to be a sense, and I was a new and the first man on the day.
At the British settlers will take a huge number of his father and the two years after the first time.
It is a great thing to fight in the story, that they're going to take the child living.
The
```
[128 tokens, no EOS]

### science_explainer

prompt: 'Oxygen is a chemical element with'  [8 tokens, starts scrolling out at generated token 57, fully gone by 64]

draw 1:

```
Oxygen is a chemical element with other cases in the disease and more likely to a relatively small growth.
In order to the most common point that the surface of the root canal is likely to a long-to-transposavirus.
- E.
-D. P.A.; Wang, H.; B., the first three years of the U.S.S.S. (J. (2009). The firstname-A.
The most as the H. "A. The second time (2020). The same time after a result are in the ‘It’s ‘N.” is only about the number
```
[128 tokens, no EOS]

draw 2:

```
Oxygen is a chemical element with a number of the cell.
- The following of the water and the brain is a variety of the condition of the disease for using a chemical reaction for the blood pressure.
- Avoidoriasis?
•
- The process of the immune system is the system, as the liver, and is recommended in the immune system is a lower carbon reaction of a good for diabetes that is found in your heart attack. When you're also need to prevent anxiety and other of the best possible.
The treatment for instance, it can also helps to reduce the symptoms and vegetables are used from the liver, and depression. A person to reduce
```
[128 tokens, no EOS]

### instructional

prompt: 'In this lesson, students will learn how to'  [9 tokens, starts scrolling out at generated token 56, fully gone by 64]

draw 1:

```
In this lesson, students will learn how to help you’re going to avoid them to ask about all, and they will have a chance to find the world’t always to your essay.
- As a teacher is the world’s thesis statement, it easier to take me to build a book on the most commonly used as a book.
A is a great idea is a good life of the main aspect of the name that it all your knowledge is a good life to be seen. In this type of the language is only by a story in the following, and the book to work will create an hour.
There are no longer have the first stage
```
[128 tokens, no EOS]

draw 2:

```
In this lesson, students will learn how to work for the student, and the other hand, writing. For example, he was more.
```
[stopped at EOS after 19 of 128 tokens -- the model ended the document]

### bullet_list

prompt: 'There are several benefits to regular exercise:\n- '  [11 tokens, starts scrolling out at generated token 54, fully gone by 64]

draw 1:

```
There are several benefits to regular exercise:
- ________URE 1.7% A review, and 3 steps in your body. If it is a result of these your child is important.
- In this type of it’s a child with the right?
- It is no longer important to be used to be made to eat to be taken in the most important of the next thing of the information, and used to the day it into the more difficult. When you can be used when you know what are on the time to read a person. This is being able to be a big role in your money. It is not always a little to help, if you need
```
[128 tokens, no EOS]

draw 2:

```
There are several benefits to regular exercise:
- ________ address the role of the use of the process
the time for your child’s work.
What Is: The answer is to the best?
•
- How to check the child learning?
In the work or help from your child to do this article How to do they’ve got them together, not have, and the information?
The most creative and the student and the school, reading skills.
- 2nd grade on your essay about any of an essay to work. How does love, please use of this worksheet?
What is my own?
The answer to a good resource
```
[128 tokens, no EOS]

### numbered_list

prompt: 'To solve a quadratic equation, follow these steps:\n1.'  [15 tokens, starts scrolling out at generated token 50, fully gone by 64]

draw 1:

```
To solve a quadratic equation, follow these steps:
1.2.7.
```
[stopped at EOS after 4 of 128 tokens -- the model ended the document]

draw 2:

```
To solve a quadratic equation, follow these steps:
1.
- Addressing the first six hundred days of the application of the importance of the field of the product is based on the user interface and a significant future for the first decade.
|
|A]
|
- The U.org/A.S.S.g.e.S.com/j., and the current of the evolution of the country of people in the war on the world that we can work.org. The project, we have a very much of the need for the first level of the city.
```
[stopped at EOS after 110 of 128 tokens -- the model ended the document]

### enumeration

prompt: 'There are three main types of'  [6 tokens, starts scrolling out at generated token 59, fully gone by 64]

draw 1:

```
There are three main types of the more information. This can make it’s possible to use of your body.
- A good idea that your children can be taken in our lives, their health.
The most exciting option is the opportunity to take place they would be able to them, and have to feel the problem to them in the way with a much for a certain words.
```
[stopped at EOS after 73 of 128 tokens -- the model ended the document]

draw 2:

```
There are three main types of the patient’s important to ensure that we often use to the body. The aim to the same step is of your car or of you are not very comfortable – or you feel like a plant through your home, and the best.
The best for the soil
What you mean when the skin?
- For example, the best step is a way that you, and the same possible.
- If you will help you have good breath your diet, or in a safe, you can then enough to be too healthy, it's the infection.
- Are not be used a simple to the correct and more difficult to
```
[128 tokens, no EOS]

### long_dependency

prompt: 'Although the treaty was signed in 1919, it'  [9 tokens, starts scrolling out at generated token 56, fully gone by 64]

draw 1:

```
Although the treaty was signed in 1919, it was a few decades of many ways for them and the American language on July.
- Why is the first reason about the government has to a good to the importance of the first thing.
The need to find more of social and the government has been known as “The first country’s Day’s economy,’s and ‘In the world” and the world.
According to be a way to the National Crime and the US countries that could have a significant role in their future, and the government is the Constitution of the most important to provide the most important to have the President of the US
```
[128 tokens, no EOS]

draw 2:

```
Although the treaty was signed in 1919, it was probably a very long been a few years of the world, and the first year, but it is no idea of the history.
“The most cases of “It’s still been at home.” – “It is that "We have been the second to consider a more time-to be.
If I want to say that it is not like you don't need to be sure,’t have. If you can’m be sure to add a good option of your time is there is the best.
- The only thing to your emotions are using the same thing to
```
[128 tokens, no EOS]

### agreement_gap

prompt: 'The students who had spent the entire semester preparing for the final examination in organic chemistry'  [16 tokens, starts scrolling out at generated token 49, fully gone by 64]

draw 1:

```
The students who had spent the entire semester preparing for the final examination in organic chemistry. (as and the "The American Academy of the school in a great time for several times, and is the first in the country. This is a few thousand hours of the nation’s economy.
- The study and other hand does a day, but that you are the two minutes to be a person or to get as a state of the science. My generation” However, it has been a world.
This is not being known by our own own home. You can do not, even do is a lot of things of the story of science and I'm using the story of the first step.
In
```
[128 tokens, no EOS]

draw 2:

```
The students who had spent the entire semester preparing for the final examination in organic chemistry.
The book provides a career and more than. Here, as well-making.
We can be done about this type of the student is another, in the internet for you can be able to the word.
This helps you like the same, the need.
3.0.3.1. You will choose to write an English language and the students will be able to check your computer. They may see how to help, the student-step for the teacher’t have come.
```
[stopped at EOS after 103 of 128 tokens -- the model ended the document]

### attribution

prompt: 'According to a study published in'  [6 tokens, starts scrolling out at generated token 59, fully gone by 64]

draw 1:

```
According to a study published in the “The first,” and ‘s” he was a few people who cannot leave an important aspect of what is not a great way to the other places (b) is to the whole way of us to the child's behavior. A team is a few years, the largest state of the way to be provided at the day. I would be known as a very clear and to keep the time we need to the world while they need to be looking to the other.
My kids is that can be able to the future of the language which this system.
I“I did not really, the
```
[128 tokens, no EOS]

draw 2:

```
According to a study published in the National Science, while the U.S.
```
[stopped at EOS after 10 of 128 tokens -- the model ended the document]

### dialogue

prompt: '"I do not think that is correct," she said, "because'  [14 tokens, starts scrolling out at generated token 51, fully gone by 64]

draw 1:

```
"I do not think that is correct," she said, "because the great to find a new in the next reason, a great thing that is the right.
The way to avoid in mind the story, and the right to make the science. The best. It can be able to read on the person is the same part of the internet to be done in the first of the top of the whole!
- How do a bit of the best way is a way to make you to the best?
The use a professional process, it is the most importantly, you have the right to find your friends, and we can experience in a school in your writing.
The most interested in the
```
[128 tokens, no EOS]

draw 2:

```
"I do not think that is correct," she said, "because my kids do they do you. I know about how much.
I think that they are the story.
My students, and my own reading.
At the kids would think your kids are going back, and how he is my story.
And all of science books is to the book and the story of all of I’m for the best!
My kids are 11: 3:30 - I am going to be found in 16, and “the great one of the math. This is a book (” she would be a book, I think I have this.
My daughter. I)
```
[128 tokens, no EOS]

### factual

prompt: 'The capital of France is'  [5 tokens, starts scrolling out at generated token 60, fully gone by 64]

draw 1:

```
The capital of France is the United States in 18th century. The first year of its efforts that the same thing, making his death of the next three years as of the number of an important role in the fact that’s, and the world will be the fact, which is more people.
It represents the child had been the world.”
Some students are in the child's work.
In this is the public health of the fact, the American government, and the school. There is no doubt, the use of the American American rule is important for other women and also is a major component of the first time of the way to
```
[128 tokens, no EOS]

draw 2:

```
The capital of France is to reduce the most often used by the United States. The project shows that most of government was so, but it was a considerable as the American army. The results of life of the war of the most prominent population was a significant role in the war.
The first thing, and the Church, you can have shown that you see this. That means that the history of a number of the Church and a look at the first time, the world.
In the world, I am so many of the most of the following of the most of the school.
My kids are a family is one of the children’s,
```
[128 tokens, no EOS]

### numeric_units

prompt: 'The mountain rises to a height of'  [7 tokens, starts scrolling out at generated token 58, fully gone by 64]

draw 1:

```
The mountain rises to a height of the north of the East.
This country: The U.S.S.S.S. While the United States had been forced to the U.S.S.S. (S.S.g.S.S.S.S.S. He is a popular and there.S. It was no one of the United States. She was also used in the first day. For this time I had a great time of the time and then they have found over, they have a long-term consequences, including each other factors, especially.
The second has also the study in a new study of a
```
[128 tokens, no EOS]

draw 2:

```
The mountain rises to a height of the earth’s way, for the air-dense. He added to the end of the body from the water is caused by being the ground as the most effective. The presence of plants are more water to have an example, these factors where the other other cases with a high resolution of the energy.
```
[stopped at EOS after 63 of 128 tokens -- the model ended the document]

### code_ood

prompt: 'def fibonacci(n):'  [7 tokens, starts scrolling out at generated token 58, fully gone by 64]

draw 1:

```
def fibonacci(n):
If the difference between the use of the best part of the case is still made it.
A-up- The other hand and a very different types of the mouth. It is the brain use of the most important to the body that the person is much more serious stress.
In the disease is a result in a problem, it in the right in the same as the same, it in the water.
One of the following of the people’s like a great way in their home to use of the world.
- "It is important for the most good and the second and people' and they have one
```
[128 tokens, no EOS]

draw 2:

```
def fibonacci(n): Some of a single individual)
- The other hand.
- The number of the most of this is that is a single point.
In this is all of the same one of this with a number of the system.
To find a large number of the surface. This is not the ground and is a bit of the difference between the same amount of the case of the same, the environment, the right.
- N. L.gunga. It’s no wonder that this technique is the best source of a variety of the most common role that is needed for the most important aspect of the most common
```
[128 tokens, no EOS]
