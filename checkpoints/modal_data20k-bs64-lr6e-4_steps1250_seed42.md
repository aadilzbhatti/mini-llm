# Sample report

- checkpoint: checkpoints/ckpt_blk128_emb256_head4_layer4_bs64_steps1250_lr0.0006_minlr2e-06_seed42.pt
- step: 1250
- params: 16,105,297
- config: {'vocab_size': 50257, 'block_size': 128, 'n_embd': 256, 'n_head': 4, 'n_layer': 4, 'dropout': 0.0}
- eval_train_loss: 6.222903203964234
- eval_val_loss: 6.195983743667602
- full_val_loss: 6.219504708248509
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
Photosynthesis is a process that is not more than the best to the right of the child to be the use.
By on this first day at the child will have just on.
The study's new ways and make you to take you make the most to the use a little.
A is the great number of this is one of the new study that we are an one of people, and you have in the study.
At your future, so it are to take at the most important to reduce the use of the health.
What is a specific for a result to make the own people, the same, we may have.
A best and the best life, is the use of its home. For a way of the student’s children, we will see the right
A, then you’t try to get a best.
So that the use:
- I need the school you“in? “” in our time a school’t make to help to
-’s that you.
The same year”
- One one day is a child to think.
- It is a work that you don’t know I’t do like the best and, you need to get just.
```
[256 tokens, no EOS]

draw 2:

```
Photosynthesis is a process that is a small level of time in the area. This is often been a way to be a bit for all people and in a different body.
The more type will also help you do the person to be a be always more common.
The child to work is a common, who is not’s just a great person to be a child, and are not to the life?
In the best of the body, and other parts, it will use of each to be a lot of the potential. It can help the end of the best, you don’t do it are one can give a new in the future that there is to do you to use.
The child would be to get us that you can have better and make a good way to a body and your child to make, they want to to the same ways.
The most can help in the dog to be more in a great-up and some the same words that a more for the children from the right of any time.
But to learn the word, they will be the most, with the day, what the next “a”? In the most, you.
If “For the time will be to be not know that to I have
```
[256 tokens, no EOS]

### biography

prompt: 'Albert Einstein was a German-born theoretical physicist who'  [10 tokens, starts scrolling out at generated token 119, fully gone by 128]

draw 1:

```
Albert Einstein was a German-born theoretical physicist who took the first and the following in this of "We is a own.
- A time, the end of the most people of the best-term use.
- the world is the body of the first of the same of the American best to the life, that has.
- If the first means.
I don“The article the first of the world and the right of The most important to the book of the life.
- The best time to the way.
- I must be been only them that we were to be to get one of the following an other ways.
- What are a few years of the child to read in the end of the new time for the water.
- You will be a need in the world’s important world.
- the Nationals is the school, and the best to the same time of the story, and the use is a long for this and the same of the “or,”'s first time,”.
" is the most important time or your school.
P. You is an long is a person to get about the people where you”? As and use?
-
- This is the end of the first example, and
```
[256 tokens, no EOS]

draw 2:

```
Albert Einstein was a German-born theoretical physicist who also have to a single-free story.
The University of a great time to the name to an “P- What” in the last, which have a important role into the best as the time and in the family. When you know not a important person to see the story.
```
[stopped at EOS after 60 of 256 tokens -- the model ended the document]

### science_explainer

prompt: 'Oxygen is a chemical element with'  [8 tokens, starts scrolling out at generated token 121, fully gone by 128]

draw 1:

```
Oxygen is a chemical element with the total and other.
The best important of the field.
It of the human and the world is a�s most known. However, in the first week, the most are a person’s new a common system to make us in a dog if we also take you make to be on the ability.
- While you also be a good children’t learn to be the new people with the most.
- Do you can know, that you can be in the life and make a problem that the world is the need for the same day.
|
```
[stopped at EOS after 118 of 256 tokens -- the model ended the document]

draw 2:

```
Oxygen is a chemical element with the potential, a fews.
What is a variety of the right source of the best-related work that is a very serious problem to know you, and we don’re more than the process or too!
In the study, and ‘As you are still about the day-based, I’t’”, in the person’s better than.’t”.
- I have, you’t have a new one of the day:
- Do I’s also be going to understand. In “-d.’s a long of your “”
- “-’s a way to make your need the use and you. He is that you are.’ll be a child’s a way of the most different-being.
- “The best’s life’s,” I’t think your day or “”’s ‘We to “’- We’s’s look a�-t a child’s “’t want for the body, the right of the child.’s good of the
```
[256 tokens, no EOS]

### instructional

prompt: 'In this lesson, students will learn how to'  [9 tokens, starts scrolling out at generated token 120, fully gone by 128]

draw 1:

```
In this lesson, students will learn how to use in the new school. For these of the number of the study are a important to be a sense for other time.
|This is not the world is one of various countries to see and use the students to the same of the children.
What who have the most, about the world has more than the first other people from the “the state’s different is more than it’s “2.
A
2.”
In
The world and most time on the following the world of the country, in the first of the most great name,’s a right of the first reason the same to have a�s the most person.
’s people the story is the most and in a very time that it is you are a good.
The key things that I be still to be a own time, you have not the best important one of the person and the day in this work, and in the children that you’t think.
In’t be about
How“a.
’s “There will be that we will be not see how. But you to “It”
The most children is the fact, or work that the
```
[256 tokens, no EOS]

draw 2:

```
In this lesson, students will learn how to be a best good way to you to avoid a lot of your-being, they can feel the most your children that that you go you’t choose a way of my.
The question, the child has the most likely, what there is more, it’re a little. It is a use,”
So you’t start you will be to keep our day to give your students. A is we have very longer to know you look. Make your time I’t’t a big. We to help to the “’s you, you” It’t
How’t’t a particular and the ‘It’s be a great. If you you not your way, but you will see?”
’s is I”
We
There can have to go, you know, I’’t’’t never’t know.
How’t do the
We have a�’t get to go to
’t need’re a dog’s the best’s the way it if I’s not to get.’t get to
```
[256 tokens, no EOS]

### bullet_list

prompt: 'There are several benefits to regular exercise:\n- '  [11 tokens, starts scrolling out at generated token 118, fully gone by 128]

draw 1:

```
There are several benefits to regular exercise:
- n.
- 10:
-20-10.7.1:
- 5
- The "-19 3,000
-1% of
- 0.S. 2.S.
- The-
-12, and,
- “-’s is a common.”
-
-
- B.6.’s
-5.
-2-m.“The 2
-s. (s) (6)
- The)).
-
- "-
- M.
-:
(s
-
- 8.5) -
- 5.
-14.
-6.
- The M.A, 3. A-3.
-9.g.||- 1.
- 4.
- A-9.1.
- The- A-5
-s.
-. "-10.S.
-15, 1.9.
- What (4./1.9:16.
-10,:1.org/1/5:
-
-6.
- the “-15.7.
- 2.
```
[256 tokens, no EOS]

draw 2:

```
There are several benefits to regular exercise:
- :
- 2-18)
- 3, A.C.
-19:10.
- A.
- 1:S.
- The following 18:
- G.
- 0.5-
- The year.1.8.1.
S.5.2. 1:0.4.4.15. (9/10
- 8.
-10.
- B.302.-01.2/3.3 =
-0.1).5. 2.0–7.com.5.10.10.4.14.5.
-6. [-2.S.9/4.4.8.4.2.3.1.4.10.3.“The and 9. (9. (4:18% of the and:0.3.5.
-1.
```
[stopped at EOS after 191 of 256 tokens -- the model ended the document]

### numbered_list

prompt: 'To solve a quadratic equation, follow these steps:\n1.'  [15 tokens, starts scrolling out at generated token 114, fully gone by 128]

draw 1:

```
To solve a quadratic equation, follow these steps:
1.
M.com.5.
If you have a child would make?
- To I be used to do they are important to the use in the first.
Cills, “in’s “F” the way” “I can make a much to love that and “’t be to”.”
The way in the time is to be a lot that your time.”
The “The “and’s”:
In the best’s more, a few? When you will see we want to do a child the day of a family.
```
[stopped at EOS after 135 of 256 tokens -- the model ended the document]

draw 2:

```
To solve a quadratic equation, follow these steps:
1.
6
|5/60-N-19.2 (C.4).
8:6, pp.org/1.1/0.5.
C.
||A was a one of 0.1.1.0.5.S.5.1).
-3.1-4/4:7.
A.’s 2.15/1.
-20.6.7,000.
- 1..1. (3).||(12-5-s 10-s, 4:|A)
5.5.
-
- The.3.3.10
||| (2. (8.5–8.
-19-6.6/19. (P.A.S. (-19.2.|- 5.9.2.
6. 3.5.org. 1.
-15-./14.1.8.10.
-13, the5 (S.10.1/0.
-2.
-0.-1
-1
-3. 1.2.0.�1. The B.3.
```
[256 tokens, no EOS]

### enumeration

prompt: 'There are three main types of'  [6 tokens, starts scrolling out at generated token 123, fully gone by 128]

draw 1:

```
There are three main types of the first of the other of their way of the more than the work of the most of the most good-like field in being, including the same time of the way of the best to help a little.
- I want to have the way to make they want to keep this way to all the key-level child’s to make that they can be been.
-
-
-
-
- If a more important of one, we will be more, that I think that you are not to try to use a best way.
-
- What are a few-up.
- Thats in the world is not, there’s one of you to do to be not a lot of a day?
- I might get to the fact, you can find a good and you need to what’t have’re the same of them.
-
-
- As an child’s I’s way you do you?
- But’t a list:’re a doctor with the answer or the next day of your idea, and even to see the work and they do is it is a simple’ll don’s to do to work how,
```
[256 tokens, no EOS]

draw 2:

```
There are three main types of the United States and the body.
If it is not to not a person's a small way to write the world, we have a very much to make for others, and people.
What is the number of the same part of some own times, what is the “I” that the story is that it a first in the most longer.’t use is the key, where we can have a time a child and a sense of a one of the the world and the body. They may get a lot of the main life to the child of the most than his day, and the students and the future and the other days.
-based, we can be a variety of the child is a home, which the same way of the two weeks from the most one of an person.
- I use of the state, you can be used in your environment, what you’s the most, but you be just more for and the ability to be a specific students.
-
- I find the first one, it to be still also to their child to do as the most for the right.
- The is the way of the students, or do, but they will think that, it can be a number.
```
[256 tokens, no EOS]

### long_dependency

prompt: 'Although the treaty was signed in 1919, it'  [9 tokens, starts scrolling out at generated token 120, fully gone by 128]

draw 1:

```
Although the treaty was signed in 1919, it’s only a�-the two number of the main level of years. This day of this of the most year to the New York was a single. It has the state in the most common than the first part of the same of the same type of a second and this year, in the other, of the most one of a “5.” I mean in what is in his story from the same year’s history, which is a person or not be the story.
The story of a few years of the fact, who the most more time of the same school to be very than his most, the first of the same.
The day was not a result, it is not for the one of the top of the second type of this time, to the same.
If the most year is not a most of the American most popular. The most more common is a best, they may really one of a few type of us, but the last, but he are not not in the same to the “up:’s to be important up to go for them at the same as, but a place in the same of the work.
The world also to the first person, you was no good for
```
[256 tokens, no EOS]

draw 2:

```
Although the treaty was signed in 1919, it was not just to be the other new year, and the most high’s, in
What would be able to find.
As us a new-“C minutes of the time of the study of the same of fact and. I had the first way of the area, so ‘I”,” and I” from the best, ‘a said?’s important to give the state in order to be to also know.’s. The first for the state of the following the same’s main words was just to a one thing, it comes.
|
•
A day?
The “F’s that it is the I’s a great time for a time:’s use by the I’s.
The way is just like the word, and a good is to the best’t use, like it, as the way, as this can help how you a very very just in it. In the most way of this way of them will be a’s’s people, and we’s more look.
If you’t get a great time in the fact. You would do to know
```
[256 tokens, no EOS]

### agreement_gap

prompt: 'The students who had spent the entire semester preparing for the final examination in organic chemistry'  [16 tokens, starts scrolling out at generated token 113, fully gone by 128]

draw 1:

```
The students who had spent the entire semester preparing for the final examination in organic chemistry. The following the most one and the best way to the work that the new time is a lot of the best than the first-term form, you make the day.
- BTt, a particular number of the two studies of the time by the country.
- What?
- The article, his world on the other of the following the use of the first people may be not a part, and that is more than all in a way of people.
- A “Ans that you can not read in the world.
- In the time and your same time and the book can be the next of the main body.’s only to be the new year, who was the first for a result. The most only the “d but the students have to use.’s a more to the other work in the following is the person a work of a person.
- What can say, they does we need to our time that is going or a more one of her in the fact.
```
[stopped at EOS after 212 of 256 tokens -- the model ended the document]

draw 2:

```
The students who had spent the entire semester preparing for the final examination in organic chemistry and the same for the highest and the second type.
If the two is a wide to be often been been able to be available in their problem. In this of the case and the world and is more than a result.
This is more different cases, which, it is just to be not been been a important, and an important to go.
The key system to the number of a larges is a�s end of the government. As not the children to ensure a child to the time out that in the body of the life the “t been created.’t not be the best than the most little as to be the school.
The way to get the use of the past the most than well in the same-time that the world of a high-based area would be been a better likely to create an more people.
The “3.
The last other way of these to make it to a few years of the following a own-quality work.
In the first of the use the way, it is a need to the own life to a lot of the number of the same for the right in the most.
- Make the own year, there has often, a wide to be to be the
```
[256 tokens, no EOS]

### attribution

prompt: 'According to a study published in'  [6 tokens, starts scrolling out at generated token 123, fully gone by 128]

draw 1:

```
According to a study published in the water, which includes the study.
In all different for example in 3, the last, we also likely to go at the area which’s use from the body where the area has been an popular world to the main. The other time of the fact, and the most likely to get a family.
The fact a dog of his people in the same person with the work, and the right of the world, and other health may get the time. The most other people are been in his children, they may lead to have a different parts in the the time of the past, and the first part of the end of the same, the most good and they is the the risk of the most important, as the world is not. By the fact, the following the United States is to a part of the two types of the best time. One and the body of the risk of the number of the same of the power of that is the world. It is a way of a variety of a single-or of an health and one of a significant research.
2. It is a short number of a lot of the best time in the United States.
As not only about the world is the study that and the number of the power,
```
[256 tokens, no EOS]

draw 2:

```
According to a study published in the time.
D.3.5. The article, the world is a important in the U. (5, and the average.25), and the United States, and is not in the new health the body levels to the data and the brain may be also so well to the use the other and of the risk to the risk (2-m) and the disease of the American-term, but is the world of the most likely to the best, however, the year is more one part of the time of the students.
The name and the study of the first of the most information in the country, we will be a good than so not to do, and the most.
The first-day the best and has the same of its school.
4.2.
1.
H.
- A 1.5, 2.
- The “3. The first person is good of the most time.’t be a time to be more as a person.
- You're the a problem, you will be a “in and for you’s a “p.
- is not to give you to’-”, you is a best’t do to
```
[256 tokens, no EOS]

### dialogue

prompt: '"I do not think that is correct," she said, "because'  [14 tokens, starts scrolling out at generated token 115, fully gone by 128]

draw 1:

```
"I do not think that is correct," she said, "because you are a more important to the other is a best.
This does still a long part of the same-day and it is a common to be a child may take more place.
The first lot of the other people that are not the one of the most. The first is a great year of all.
The first way of the end of a great time, and it is the body that are still the time for the world.
One of the children will be the right to make how you’s not be at their most longer in or or our own, and it is it does to be the students in other people.’s a type of the use in the most, and they’t go to the most the best of a good day, and a way is the same way of his way.
1. The day is the first of an child to the “It said I I also to know me in which I’’s use for me of the same people and?
The best and is the way to do you should have like. They will try, to have a lot you to the people to take the child to create a right the way.
To do as it“The time
```
[256 tokens, no EOS]

draw 2:

```
"I do not think that is correct," she said, "because we really to see that it is much that they need to remember.
The “’s ‘b., we will be an long of what the I am. Here and we are a�t are the best need.
At
This is a “of-I’s to think, if is you can the I would be able to know that the only to see. It’s time. The question may say that?
" do it has about you have a�t be to have a number to consider from the end of a lot, is the “and is more ‘and”’s a ‘and’’ to be made to be,, of. The “If many is the day, the person”
The’s a way to help to find a way, and the other and can be know in their way?
’s important?
’s’s a�s I’’t know the use with this.
’s this of’s a year to be a person, to be to be a good way to be more. The way of your own example to do you, an problem
```
[256 tokens, no EOS]

### factual

prompt: 'The capital of France is'  [5 tokens, starts scrolling out at generated token 124, fully gone by 128]

draw 1:

```
The capital of France is a good to the main research to the following the most of the following the following. The world is a number of the United States which is still used to the United States and a large role in the United States, but we have too.
With a key health systems are also for the public as we will be a few, but you would need to help their life.
How
This the best time of the time if they are able to the most more. It is being found as that it does not some way to see, but he’s first and you.’s be a time in the book, it will can be them to make this.
If he is a good of the best idea of the time, they will can consider.
The next’s very than a child may put, that we get you think you to make yourself where some a child will keep it to see any time (d" to read, which are the other or get us, and is I can see the book, but the way to be still to know that what you are the world. Here is to see what you are a day.
The article, a best day, you will have the best that, they also a person’
```
[256 tokens, no EOS]

draw 2:

```
The capital of France is the American "The same part of "In the United States).
The British Department of the most more in the United States has a study of the second people, as a different, including the age of the American.
Pental system is a high-based-called water.
-
-
- The first number of the data, but it will be a particular types of a long-and. The body of the last group of the world, the same study is the work.
- How you will have a time?
What is any time?
-
-
- How can do the more or you do?
-
- The the next’s more than your-making of the “It’s’s to be a lot of
-t not have you have’t start to the most’t be the best to you’t’re know, if you’t use you
- As a example, but you to make what you’t go. “It?
-
1.’t say?
- When it’s’s be a important variety of you”?
- I do? This is?
```
[256 tokens, no EOS]

### numeric_units

prompt: 'The mountain rises to a height of'  [7 tokens, starts scrolling out at generated token 122, fully gone by 128]

draw 1:

```
The mountain rises to a height of the the first.
P.
Bain and a “The country in the last year, “a is also be been the day for a few years, and the same area of their own time and the early other-like, in the best time, and with a great world.
The first day will start, you.
The very good way to see this time.
As the following the same-
-
- I will use, you want to look.
-
- “You can do you.’t know that I don’t need or take your body.
- I may have the home and all to have a way how and just do you to take the your look.
- If you will’re’s need the
-
- You’re get, it the people!
- You is going of it’s you in “or’t be one’”-term-f and the life.
- To you know you you’t want to’t take to
s
-’t know?
-time
- you to make you know in the other students who have the-based�
```
[256 tokens, no EOS]

draw 2:

```
The mountain rises to a height of the city. While the first of the most way of the top of the most than the case of the following the first of the second-time and to the same body can be developed to a high-effective role of the same time to the past a lot of the work.
What may think a time and for the first.
|
It would be an few, the same part of the past of the following the world.
In the first of the main life, a particular of a first people, including the child is they can show about the “by of another of a time.”
I could be the I’s more than the following at the “You can also.’ll the very child that the one is your best to learn if you need in the need to find the I’t only a way that they will be to try.
There.
The right on this study and the fact and the way.
As you are a lot of a good’s the way by the students can be the world of the most thing.
’s a child’t get a lot that you get the right to work and have to take the next to get them.
’
```
[256 tokens, no EOS]

### code_ood

prompt: 'def fibonacci(n):'  [7 tokens, starts scrolling out at generated token 122, fully gone by 128]

draw 1:

```
def fibonacci(n):
- D.com-Tas, A.
- E.S. M., The study of the study in the the study, and the New York, “s
-’s, they need to get a small of many and a more for the same side of well.’s a result of the same number of a way or is a short work in the child is a number of the United States.
I’t going to this is a few people that you’t want to be been to the students to be many health. That’s a bit of the life and a major health of the following a result of life, and in the same.
The most new people, then not the same and it’s use,’s not used to be more. These is a little time of example, who may have one’re good or any example. The type of the age, there is an lot of it’s a more to get a problem for a new study.’s not be been have been on the case of a child in the day of many and the world. The book.
The most of the same way of one of the world, there
```
[256 tokens, no EOS]

draw 2:

```
def fibonacci(n):
- 22.
- The B.
- D. D.e. (F/5.
-10-/18.8.0.
-1.e.2;
- G. J
- G.e.0. (10
- 0.g. 5.10.-.1.,
- 2. The "/16.||- 1.
- 3.C. (S.8. 2, pp.
- A. (3.--.10.1. 1-/3. (4.2 (25.2.14).
-9
-
- 4. (1.
- The.1.3-4.-15.2.
- 5/100.-6.| (2.3.
- 0.5.1.g.11.5.
-
- and- 3. 10. (4/d,,
-1.--
- 6.
- 9.
-/15 (11.0. The-S.2/19. (2/3/10.16-2.
-19.2)
-
-1)||
```
[256 tokens, no EOS]
