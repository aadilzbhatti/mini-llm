# Sample report

- checkpoint: checkpoints/ckpt_blk128_emb256_head4_layer4_bs64_steps10000_lr0.0048_minlr2e-06_seed42.pt
- step: 10000
- params: 16,105,297
- config: {'vocab_size': 50257, 'block_size': 128, 'n_embd': 256, 'n_head': 4, 'n_layer': 4, 'dropout': 0.0}
- eval_train_loss: 6.735891783237458
- eval_val_loss: 6.716515302658081
- full_val_loss: 6.735948399446362
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
Photosynthesis is a process that,. more than the world of the right of the world to be the early longer.
The water to the water of the first than independent.
To as the same new.
One is some. It will for the study the case a the most the number, and the right and then the second week of the potential.
One and the amount to the government, it, the study with the following time.
The time of they is at the most and other, like a long the health and a lot to protect the following children, which’re with the whole is, when it will.
A's that the word. The story.
In its than all for a clear.
-d, but, the National, it to show will help, is also so,’s’t a day in the the use it know some people can be used the a good States, �’t in our time a day to be as they to their the family.
I
We want.
The end”.
We are one, the same of our the other, but’s to the “What a good, and they’s as, there, because that has at the case
```
[256 tokens, no EOS]

draw 2:

```
Photosynthesis is a process that. these family the time can in the following very number to their very the way, and the water a strong people be in a different is to the a include a be,’s thes than a be that people the the they�s in the number of a are the most is not’s’, the world or a‘, including most. These, and the first, in order the body to create those’s the use of each to be in the body of the time of the a number’re this, so. These children, or been to be a problem that the body of a little impact about the ability!
Why.s other, would to understand any”
This have it is ‘s of the “M- It to the right.
’s the first was to go to the same in the book, which, in the body-up and other the environment,’s in the “I that need of the story is to the next one are can be to the use the case with the use, as there “The need to the name as you is the “- ”, and then are found this are not know the end and has
```
[256 tokens, no EOS]

### biography

prompt: 'Albert Einstein was a German-born theoretical physicist who'  [10 tokens, starts scrolling out at generated token 119, fully gone by 128]

draw 1:

```
Albert Einstein was a German-born theoretical physicist who to the first and the two land. When the city of the same- War that, as the United of a. The first of the European-sized�to and the the world that is part of God, a group of the future.
One. The future of city.
A and the first of the American U in a small to the the U Empire of their the first have found in the war.
The first that it was considered that the USthian, the North of the same and in the island and was a result to be to be already the United States was a long-
| What from how a group of the water, in the body is the first new health for the next to the early areas to be of the last and his city had the body, and the area the National and is of the main and then the last is the first not is they had the United States is a long the population and the same as the “The only not from the first likely to be not the “that into the world when the “The next day of this is that he is an,’ It” and theirll be the person and and then no other and a the right, and have that was the most and
```
[256 tokens, no EOS]

draw 2:

```
Albert Einstein was a German-born theoretical physicist who and of the British and for their own of the country.
-made.
The name to an was a lot for this health, in the last, which have a variety of the country for the other by them in their family of the best. The fact to be used of its been the other, are all of the public.
In the main of most for the development or been the last time.
In a result that-day. The way to the same people of our the first and the first new a common cases to make the U (The current, is found been the next into your students in a few of all of two, and the best and the brain to make they are a great with the most.
In the amount. The following new,000 school.” and make they is considered the world of the study for the same in a simple, which.
The process of thiss in the future.
One to their more things, it if this is in the state, we can benefit of you, and we’s, and others with your “It is that the entire child’s time, we’s different-hand, but that the amount for the day, they in the brain types
```
[256 tokens, no EOS]

### science_explainer

prompt: 'Oxygen is a chemical element with'  [8 tokens, starts scrolling out at generated token 121, fully gone by 128]

draw 1:

```
Oxygen is a chemical element with,
This, is a great when it the data. He the last, and, but which of the new-inThe first important to the only, we was the how it is the most. In � more. This�s of the state the next or not your person’s time the most of the question that. It, so out the last and to the most-the ‘ first that as a home of as as they is by as is the first is.
The “The best has as.
The past, this,”s, and the first.s the school, the use the � would.
|
The first time of the same and have. If it’s of the most” is been to the same, the right were the first them is that the same.s you’s the world in the study. It are a lot are made. When that are a good.
“-term time to learn a story that”, these one of a high” to a child to a variety to become made of the end for all of the �’s a little factors. At the same people is the “the state for your business
```
[256 tokens, no EOS]

draw 2:

```
Oxygen is a chemical element with be is the world as a own that to be, but of the world and the U, the United States, and the most time in the whole of the other, and some.
In all, there is a process of As this. the environment from the right the area to have a more be the most of least been to be been in the story, the world and in those, the same, especially, this year and the development in the whole than the most is no other other and other, the story who the best of one of the most and the day in this work, the state.
-
In it. The.
In the name and the following have the most. When.
’s “Now will be that we will be these it had at’s the last year, and to take as the story to a good’s more point on them” was it’s “-and’s the the most.
- The
-3 and a part, and to be not they did on the word the American country.
’s there is more, the time to the best a great, or in the world, you know any people’s we can help
```
[256 tokens, no EOS]

### instructional

prompt: 'In this lesson, students will learn how to'  [9 tokens, starts scrolling out at generated token 120, fully gone by 128]

draw 1:

```
In this lesson, students will learn how to that we.being by the their the a good, this, and, so a week to.
A
In the risk.
The same in a big. We to the world and the the following been at
The first time is an to your other of their week to be a lot, a particular at the world on addition out of the process process.
One) to the first to get the first time of their the other health, and health is a great is one, we will can have a new, a better have the best of the children to make to a result has to make.
One.
How is the project in the dog. The following information. The most or the story to the importance, it.
-being to the most issues of the study who will help the environment it don also of the key and that should have to get you is that you who are a result, and will not a single information.
If it were more likely thatThe body. This have a lot of your students.
If they’t of the first child the most and others of the world that may cause or a the same and the brain information you of a computer, we is important can find there.
- Have it have
```
[256 tokens, no EOS]

draw 2:

```
In this lesson, students will learn how to a doctor.being and is the only small be�-being,”.com
 of time and a lot,” are to be a have”: It.
Here to thes is the the is the question.s
The children it are a common means” a certain the most use a clear,“the
As the most –'s, and a healthy likely is in the the best that can also, you.
-in-A
The example to the a different the a good for the two-up and a new for their. Once their you are used to an “- How can also are a range that is a time by the same year, are one of the life.
When’s is something it they are the same time, the best that do the “see a way the world with the potential and of some can be a few work was able.
On a lot to do it may”. In as they can also which”.
The best as a single.
- Have,, because the “or.
The body – is an old in the way for the first, we can would be can have to see a new, a week
```
[256 tokens, no EOS]

### bullet_list

prompt: 'There are several benefits to regular exercise:\n- '  [11 tokens, starts scrolling out at generated token 118, fully gone by 128]

draw 1:

```
There are several benefits to regular exercise:
-  ( �
-
- The

 and a
- How: The most common--ay2:|||:// (
-
-, but. It is a key and the most from available the most. What from a single- in your year to start and the main important to find to know not you,’
 It is some, it is the next role of the U.
A. This’s and can have the new, these and learning. As for the following of the study that the first important can be the most and its to be more is the risk.
-year.
In your common. It are an appointment, and do the most-3:
-being.s a few is the “- In the process.
Fers. the state of the “I the first work of the ability of myt are often the children, to be if you’s from an in the time of the same time by a long for the best for the best, which of the last of the state. However on the most.
We are, a more, a few things in the case by your dog to the two. There is we”.
```
[stopped at EOS after 251 of 256 tokens -- the model ended the document]

draw 2:

```
There are several benefits to regular exercise:
- :related

 in the
- and-
For the second, or a computer and a
Anothers will also are no-
- The ability as a doctor for your child of that is a this children or the following be a new the water of This of your child and a child of the last-based.
- A health of the best is used to get. The main are a more of the best or what the most- Your ability, you is a high and the process and the work.
The use it should be not they know no and you think, or, you the problem
-being?
The same life and the rest to get been the current a good time and it with the most.
- Do they can be a regular-up.
A, but to our diet from their and their the study is at the potential?
-related health of the body. It of the same of this and how you and the child as your family, it, the first a variety on the work will have a high and well of the first own of your own and an will be a simple. We will be become some people for the time, including the use a few is.
These have been your child to be
```
[256 tokens, no EOS]

### numbered_list

prompt: 'To solve a quadratic equation, follow these steps:\n1.'  [15 tokens, starts scrolling out at generated token 114, fully gone by 128]

draw 1:

```
To solve a quadratic equation, follow these steps:
1. If that”
What”).t””, a high by a high:5 with the process –, including
G
In a way.
How to help be anemia, a very some to have the way or not they can see to ensure and a study of it should do and well, a list.’ve can’s will have on an opportunity a large’s time between the amount to, is a little of a week,,000 in the second.t can be it to the same or been the first you at the most are the.re, and other in the world to explore, their own to be the use and to keep an opportunity and may be even a year.s to get you get to the fact, a lot for the future and the need of the future for the amount, but at the same health them, which and the right and if a new of the most way of a more in your article about you.
The most, he.
One to achieve that can be no, the top or the next, and their to get a week to see the people are a child is it is a simple is the end.
Another school to explore the work, in
```
[256 tokens, no EOS]

draw 2:

```
To solve a quadratic equation, follow these steps:
1. With, used and the right. the way,, of the people time of the “You who is a small, we’t” from the most, a day, and, the most.’ our more and some have to the same than the “I, have going.”, it of the most”“We be this’t is the main, where our ” a “the world of a new “The world and the book on his “The idea.
This””
You is in his day to do the idea and I or the same than the most of this to your of time that does of ” to the other is, which the people should be able in the way.
The and in the United”, with we find it have a person of the same common time, whatWe’s the most life, and how I�When and the following look of the main of the, and it of the same things in this, it have to work and our the right in the most and is the own of the first to the world of ‘ the “I
To be not I think in the children can be a number.
```
[256 tokens, no EOS]

### enumeration

prompt: 'There are three main types of'  [6 tokens, starts scrolling out at generated token 123, fully gone by 128]

draw 1:

```
There are three main types of the second-the a well is in two as addition and we have a research. This are.
As of the year to and social and improve the most of the main- The
What, you might be that is so the, and they can be as a bit and then the first than a other important of the way with the risk. This may be the future way to meet to find in your than your and the environment to protect you are more diet with a person or not you is and your health to keep you. Once it want as the body your the health, and what is to be to be very, but to the work from the body of the way for the body's a result to the new types for the ability the main. The problem.
Do this kind them to be a few of an effective, or a list?
or and so how by the most of the children is a week, they may be used. You are a big- If a single. It.
-
My?
It to the need and then a person to manage to be important.
-term.
The same child, the other types in the same and the work.
-
This way,000 research, you is designed to the
```
[256 tokens, no EOS]

draw 2:

```
There are three main types of They it can little of the other century’s a lot to’s, would be they can be this to it.s can be a new of the city the world of the time to the United States, the United States and the world of the first in the � the first of his the first new of the first have they had a very, the United States.
In the United States:
The
The name, and other, and the same and the first only he. The first a bit to determine the following be the name, the city that was the same time (The following can take as we are the future, the first are about a is it’s that it had the family”, the �The war.s in the word, we could to the time,,.
The way is just the �” to the largest is used the best” a group, a place, as we to know as this “’s.
What in it’s the same of thist” and was a long world will’s of the people we’s was the day of the one it you was not than a more person’s, the question of their
```
[256 tokens, no EOS]

### long_dependency

prompt: 'Although the treaty was signed in 1919, it'  [9 tokens, starts scrolling out at generated token 120, fully gone by 128]

draw 1:

```
Although the treaty was signed in 1919, it to the only the York, and the city,.
the War and an�the the the time and “The first-, were the way of
The father that he in his the time of he as seen and the world of the right by the country.
In the “or for in the other,”-on, you the name.
```
[stopped at EOS after 75 of 256 tokens -- the model ended the document]

draw 2:

```
Although the treaty was signed in 1919, it.
He of The 18,, a
’s.
The American of people by the government of the American study. The the National States. In the world.
-year War of the first, one and South.
C�, are a unique the first, and, and the most.
A, who was the first, to the most of the only the � an individual be the same the war, including the two of a high, the other of his last and then made and a world of the same time to create in the early common, he is used to our his the name to go to a country of her in the the most-hand of the use as an�B.
3’s about the country of this.
The University of the most likely, and the United States is the “-scale and the National most to be a great of the work of the French right of the “The country.
””, it is not a lot.
The city was by the number, who
The state of the world, “You would that not the children to ensure a.s and a more life in the Americans, if the “The new, the new
```
[256 tokens, no EOS]

### agreement_gap

prompt: 'The students who had spent the entire semester preparing for the final examination in organic chemistry'  [16 tokens, starts scrolling out at generated token 113, fully gone by 128]

draw 1:

```
The students who had spent the entire semester preparing for the final examination in organic chemistry, and with the most and what the last (a to be of the state to, and his and the use, as the most-
You of the same-time that the the two to produce their way in the same and the world of the study for a result of the most common.
-based countries in our least than more to the following to a few years of the following a the following than the future and the right, and the use the following be it the body, the most.
By a lot of the number. What of our people would in the most.
-year-
As to make the world, a wide to be to be the way of the country of the study.
In all of the water in the most, so as the most- The water are the area to the same, and are the body of the name for they had by the most- The next for other system the world to a little time is a number with addition.
The United States. This, we a lot of the most with the work, and the right of the world of the other health a
A (- The University, has the city in his children, they may lead by many can be also in the the last of the
```
[256 tokens, no EOS]

draw 2:

```
The students who had spent the entire semester preparing for the final examination in organic chemistry, while as.
We it the study, as a new the-based and his first is the the most important the people have as the world out in this than to the most the are could. “The world are, but’ be the to read.’s, especially.
It were an important for the United Statesas’s the world, so is a way of the next,’s than of an is to know the new to share to look is their own to become as that that people the world, and the first the most issues, they can will be is a few, there that are the same of people. This the time.
If I’s the same children.s, there is the world’s of the world’s.
’s interestings the, is not in the new, the body to a variety of the ability to have all, so the first, they may also and time from.
We can be not the book of the main for the people them to go, but’s of the most’s? The home, the way to be one, which will not with others, but,’s do and how in the
```
[256 tokens, no EOS]

### attribution

prompt: 'According to a study published in'  [6 tokens, starts scrolling out at generated token 123, fully gone by 128]

draw 1:

```
According to a study published in the main- the same and first other in the most and the world of these? The last and the United.
The first-
- Use and has the following�The data.
- It is a person, there is as you, such.
But the the world and the work, if it were the first.
This way of the most time.
According to the future.
If more object and the world that if some difficult are a problem, and it are a “in and for the right and it is as a new, as the use for the most and important’s what the environment, is on these data, which should be to make us at a few point that is a large energy.
The same life.
The the same health of your own time, the “What”’t can can be a lot of the other people that are a” (We can cause to be you’re a problem.
The first way by what to be a great- It of how a strong.’s the time for the world.
One.’s we’s”, our verys- Do it have been the “or.
The best
```
[256 tokens, no EOS]

draw 2:

```
According to a study published in the first this large.

. As to take used is the “The time, of the story of its a sense, are you.
The most the study of a big for the National for the early part.
What’ little when the first than a series from by the body can be the “It a good
The time or taken in which are the process, as you�e and of the same, and individual of the best), which in their own of the number of the water was the people, which have that is you” of an organization have a major of the right as in the first. The risk of the first.
The first, and are it is not a book.
According.
The first time with the number of the work of the ability.
```
[stopped at EOS after 165 of 256 tokens -- the model ended the document]

### dialogue

prompt: '"I do not think that is correct," she said, "because'  [14 tokens, starts scrolling out at generated token 115, fully gone by 128]

draw 1:

```
"I do not think that is correct," she said, "because this time and the a very body in the world, that do., the right,.
At
This,
The whole one in the following is a good of the U is in the the main, is still States, they is only day. The United States, and other than the state,, and the world. The largest,, which have a series of the region have a number to the world for the body in the end is the whole to make a more in the beginning to the “The most,’s, and it is a more “If, of the world, which have from the world of this is in the day a unique common or they can also can always to find a way, and the other and in the children for the same than to the water.
””, which the best, the same years-
Another little time, the city with this. He the study of this of the the right and are to know at’s to be also and other, these.
For a lot, but to know. In your own the most, a big and are to take not the most of a way a “I, but a number, and the right the world of
```
[256 tokens, no EOS]

draw 2:

```
"I do not think that is correct," she said, "because and it is a single of a more and so, or a significant time you, as.
With a key up that are also can have your little about this is a doctor, these things to be the time.
The first year and the work is done of year time of the fact is the most of his your own to be found to the best in the world and its not are found to go in the first and therefore. This is an excellent.
The first common, it that can be the use,’s, he is a good”: So of the time, they in a woman.
-being. That, and we will have a person could be more as you are important that should seem their system a result.
For the best, including the same problems to “the first, but that“s important is some-’s of the same way, if the people are an important and our own or you’t are able.
In thes the other-19 with the two-time the firsts they are able to thes “b://, and the same thing in the best in her day to be been the study, but to consider whatThis- What see a range with
```
[256 tokens, no EOS]

### factual

prompt: 'The capital of France is'  [5 tokens, starts scrolling out at generated token 124, fully gone by 128]

draw 1:

```
The capital of France is the second, to a it. be used as a bit even of the most.
- What is.
il.
-.
-
According on December al.
In
- L have to make the fact for a�t think and a large than “We’s”, the same to make the is a question is to the main of the time? “For any, he. This is that the the law can also to this they would do with in these of their.
You is not are what the day.
The the the time.’s there to the same life of the first, the key and then you have not in the best to the most’t said the same to me, and you’s we had to be as the own years to the environment.
In the way it was they in the other of the best “It?s the country to be in the the end“In the fact of the most of any school. In the next or the best it is a great of any and and most?
And the ““P.
- I must look of the way as one. The most-like, however. I is
```
[256 tokens, no EOS]

draw 2:

```
The capital of France is a time.
The different is, and how to have the time and the early other-based as the most, that�- A
When
-of- The first only for the body.
The US) by an is more part the research.
-. The World, the process of the human-term) such on the first than that.
An and a result in the main for the first the best as their number that could�The name of a article, to be a way to be and all, to a way.
The first time to the entire time. If the next for this article, or a group to the amount.
-year.
-risk of the time of the first in the same-19s of the most information, making in the last conditions.
It the United States,”-term- He about the first own is a large.
- The “The environment of some, the own to support the same than you have that can be the people the first are this and make you’s and the end and the first own-term a good people could be that’s.
The most change.
As you are not to the brain of the best, we
```
[256 tokens, no EOS]

### numeric_units

prompt: 'The mountain rises to a height of'  [7 tokens, starts scrolling out at generated token 122, fully gone by 128]

draw 1:

```
The mountain rises to a height of the are a other tos the first, but to the only, we the time will is.
In the state of to him which have the way is the United of
As we to.
It, and then a the same.
The most- of the following the and the way the use have a long of the most.
- The the same, including the world.
After the most by the body and then been much of the American, so a well, a woman, and when the same than the following good-day.
The world.
The last the government of the the United States of the world, if you of the right, who or used of the same system by how about the health for the United States.
There.
The and their not that can be not and water and the body which are it is used of a variety
The process.
```
[stopped at EOS after 182 of 256 tokens -- the model ended the document]

draw 2:

```
The mountain rises to a height of "One, a similar. the state of the world. In a new a wide of war. As and the country. The first, and the city and have the new, with a large-year-level and later, the United States were still have a way.
-
In the state.
The new of the number in the the next, in the United States, “s they is more were that” on thiss by a new and and then more people to the government in the idea into the most:
’s the end; which can make the largest in the first than the most. He been the most.
I to the right on the “It know a new on the study of the end of the first people a better these, and the work the other, but to the first year, we are done of the following not out the best, and in the same. If to make.
The most, it will cause in an�and.
’s, but in the best. These be that is a person is also who may have one’re, or any, but is to give that is the first in an object.
The next, it will be the power.

```
[256 tokens, no EOS]

### code_ood

prompt: 'def fibonacci(n):'  [7 tokens, starts scrolling out at generated token 122, fully gone by 128]

draw 1:

```
def fibonacci(n):
After the time, on the same the ‘�In the case of the U in the � of “or) on a �.
- A ofs the National“"s the last was still a-2, this and a new of the United States-, a series, a lot, and the first part.
" all time at the city has, which the end.
A;
In the area is a small and a world. The U, the time, when an it would be found to a new world with their, and use to the people are not the following in the world of the way of the government to the National-based,, it could be a specific energy from a year for a number.
- The last-day, and the other people had at the children to be not the time, and the children be no or a person are more than the same, the development it must lead is able, and other, which can be it must be the water of the United, they they should be the best energy. They are an important to be to be a whole in a result, especially a be they has and the data your,, or it are considered in the same time.
-
```
[256 tokens, no EOS]

draw 2:

```
def fibonacci(n): TheRR.
�B (� of0. Thea for the Greats of
�, the same the Rian. The last for a "19. the three was the University of the state, and the own of the common-making about the world of the war the the "-in.
-to. S.
–3.,%,.8.9.
-1718) of the
- (e.4: The
The p.1""L.
-19.
```
[stopped at EOS after 109 of 256 tokens -- the model ended the document]
