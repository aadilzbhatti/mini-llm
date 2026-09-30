# Sample report

- checkpoint: checkpoints/data10k_80k_emb256_lr3e-4_seed42.pt
- step: 80000
- params: 16,088,913
- config: {'vocab_size': 50257, 'block_size': 64, 'n_embd': 256, 'n_head': 4, 'n_layer': 4, 'dropout': 0.0}
- eval_train_loss: 5.0538554191589355
- eval_val_loss: 5.347294759750366
- full_val_loss: 5.277847030017275
- max_new_tokens: 128
- seed: 1234
- block_size: 64
- device: mps

Generation stops when EOS (50256) is sampled: EOS is a document boundary, so text past it would be the model starting a new document.

Context note: the window holds 64 tokens, so with 128 new tokens every prompt has left the window by generated token 64; everything after that continues the model's own output only.

## Sampled, temperature 0.8, top-k 50

### definition

prompt: 'Photosynthesis is a process that'  [6 tokens, starts scrolling out at generated token 59, fully gone by 64]

draw 1:

```
Photosynthesis is a process that can help to improve overall quality and prevent your health.
- In addition, focus on how can you work up with your goals and goals.
- Use an essay of this article, and the best way to explore your overall health, support your health and wellbeing.
```
[stopped at EOS after 54 of 128 tokens -- the model ended the document]

draw 2:

```
Photosynthesis is a process that is used to treat the same kind of bacteria in the field. It is very popular for every given time in the world.
The reason for this time a person with a different person to be able to get rid of the material. One thing is a person in the human body as that the person is too many people are not going to be from the person.
How can you know about the difference of a person?
This is true. It is really a very great way to tell us how people choose to be aware of what is it. This will go to the person to do for and to be the most likely, and
```
[128 tokens, no EOS]

### biography

prompt: 'Albert Einstein was a German-born theoretical physicist who'  [10 tokens, starts scrolling out at generated token 55, fully gone by 64]

draw 1:

```
Albert Einstein was a German-born theoretical physicist who had chosen the first major author of his work and of the poet.
The artist would be an expert for anyone who came from the same day as the British Church and was born in the 19th century, and they were the first part of the world of his life.
We have been studying in the Bible because we have seen the Christian History of the Roman and New Testament, but since every time, in the time we see it are a wonderful and an important part of the world to the culture.”
The English translation of the Greek Greek word is for each word the Hebrew word in the ancient English word.
At
```
[128 tokens, no EOS]

draw 2:

```
Albert Einstein was a German-born theoretical physicist who was developed in a century by the French-European Civil War on the East. It was then a very long-standing event, and the political force of the Catholic Church.
The Church states that the Chinese Church had some influence on the British society, but not every religion. Because of its most serious in the United States, the Jews are part of the world. This is the church. The most influential, and the Old Testament is written to the King of the world of the U.S. Constitution, the American Civil law. The same is named “The Day of the United States of America,” in the United
```
[128 tokens, no EOS]

### science_explainer

prompt: 'Oxygen is a chemical element with'  [8 tokens, starts scrolling out at generated token 57, fully gone by 64]

draw 1:

```
Oxygen is a chemical element with a system of the air. The high level of the electrode is composed of chemical gas that uses a chemical to detect it from the cell.
As the gas evaporator takes to be removed, the voltage becomes relatively low. Some of the most common components of the ammonia. The chemical reaction uses the main oxidation potential of the hydrogen atoms, which is the main component of the hydrogen-oxide that is the main step in the electrode.
The electrode in the atmosphere is the top of the electrons, which is the most common type of the chemical particles in the atmosphere. This is also a part of the hydrogen atoms of the hydrogen (1
```
[128 tokens, no EOS]

draw 2:

```
Oxygen is a chemical element with a chemical function in the brain. The term is one of the most common causes of this problem. The cells, of which the cells are also seen in the tissues. The cells are the first of the teeth, so it can be found in a specific form of bone tissue, which is found in the mouth. This may result in the age of the teeth. The result is the condition that the bacteria or cells are not allowed to control the tissue in the body. This can be treated with the ligaments and a mild bone that can increase blood vessels. The pain can also be made into a hormone which is associated with anemia.
```
[128 tokens, no EOS]

### instructional

prompt: 'In this lesson, students will learn how to'  [9 tokens, starts scrolling out at generated token 56, fully gone by 64]

draw 1:

```
In this lesson, students will learn how to write the paper and have both a simple essay. The first step is to describe the role of the essay on which the character has become the most accurate and accurate essay in the essay. This essay is a great deal of focus on the essay - a story that the thesis 's the best and how many students have to do it. It is a better understanding of what the student will understand what the introduction are. On the other hand, the teacher will explore the most beautiful and the world of writing in the literature.
For a long time, the students are able to write their own ideas and how to express their ideas.
1
```
[128 tokens, no EOS]

draw 2:

```
In this lesson, students will learn how to write and change in various paragraphs, and you'll be able to communicate with others.
A better understanding of the topic. If you are already having any information, you can make an easier choice for your students.
When you work on an assessment, you can see in any way that you’re using your students. You want to spend a certain type of assignment and help you learn what they’ve been doing.
- To help your students find more about your knowledge on how you do.
- There are two areas of the class in the book:
- How to Write a link
- How to write
```
[128 tokens, no EOS]

### bullet_list

prompt: 'There are several benefits to regular exercise:\n- '  [11 tokens, starts scrolling out at generated token 54, fully gone by 64]

draw 1:

```
There are several benefits to regular exercise:
- ________ Asked Questions
- What are the most common dietary benefits of sleep?
- How do you eat in your daily exercise?
- How many ways do your child feel comfortable?
- Do they sleep better and more?
- What is the right and wrong way for your children?
- How do you take your school?
- Do you know what you want to know more about them?
- Do I start up to get a day in a classroom?
- Do I get the day in the first day?
- Do I say I want to go it?
You need to keep me healthy!

```
[128 tokens, no EOS]

draw 2:

```
There are several benefits to regular exercise:
- urn: The root of the muscle is the most common.
- a mental health is a typical physical device. There are various types of arthritis:
- a person or a feeling of having it or a physical organs.
- If you need a physical disability, you may know that they are not only aware that you’re experiencing a significant problem:
- a medical specialist should be a substitute for this or next.
- to be involved in the treatment of an illness.
- to take a positive, look at what you say, is the best thing that is your job.
- For example, if you
```
[128 tokens, no EOS]

### numbered_list

prompt: 'To solve a quadratic equation, follow these steps:\n1.'  [15 tokens, starts scrolling out at generated token 50, fully gone by 64]

draw 1:

```
To solve a quadratic equation, follow these steps:
1. The correct answer can be used in the
Fobiral form of an oscillative model in the form of a hacre,
a. (i.e., the t-matella + m) and the d/p (d) for a specific and quantitative assessment (a) to identify the underlying cause of the N. c. h(e).
a. is a case of an experimental trial or a group with a single-oxic response to a type-carolobin-in-deth, which shows that the nocardiosis is a chronic disease that is found in the form of
```
[128 tokens, no EOS]

draw 2:

```
To solve a quadratic equation, follow these steps:
1. The following point to the table.
3. When a point of a table in the reference period, go ahead and the beginning of the page.
2. Calculate the number of notes, and then the formula is not enough. What does it cause the difference between two, and more are, the result it to be, and the same, such as the
what are the length of the URL?
When you want to be the first part, you can use a letter of the text. You can see if one is on the column. When the password is correct, you can use all the two or two different numbers
```
[128 tokens, no EOS]

### enumeration

prompt: 'There are three main types of'  [6 tokens, starts scrolling out at generated token 59, fully gone by 64]

draw 1:

```
There are three main types of products used to get a significant problem. The results are not only used to support the physical and psychological aspects of life, but they need to be found in a variety of settings.
As we are concerned with the results it is a critical aspect of the environment to keep our lives out of our lives in the long period.
The more important things, the problem of which in turn can come from it. One of the most crucial things to remember is that it's less important to remember in the future of the animals. They are known as the "normal" of the country and if they are not familiar. Some of these things,
```
[128 tokens, no EOS]

draw 2:

```
There are three main types of chemical reactions: one with a more common part of the chemical reaction. It is the type that is used to represent a specific reaction and a physical reaction.
So, how does the body make it?
The root canal is not properly taken, and if you have the same function of the cell, the body will be from the end, or the lungs are not only a problem for you.
What is the process of thinking?
What is the body condition of the cell?
A condition is caused by the individual organ. If the function of the function is based on the same, or the body may be left out of
```
[128 tokens, no EOS]

### long_dependency

prompt: 'Although the treaty was signed in 1919, it'  [9 tokens, starts scrolling out at generated token 56, fully gone by 64]

draw 1:

```
Although the treaty was signed in 1919, it was the first time in 2017. The second city was one of the first-year-old and most famous artists, which were named after the British administration in 1948. The fort was not a way to reach the military, and the military in 1743, it would hold the nation to take up until the British Army, all a few years ago, including the Chinese Indians, with more about the highest-reaching problems.
The government eventually is not necessarily on the EU government for the first in its first place, but the best means to be the only company in all the world.
In fact, the United Kingdom and Canada have
```
[128 tokens, no EOS]

draw 2:

```
Although the treaty was signed in 1919, it is estimated to be a part of the new, the number of reasons for the law of war. It is not likely that such law is not required for the federal government. Even though the federal election was a criminal law, the court should be made of law: a criminal law and state the constitution of the criminal justice system by law.
In the election of the Supreme Court, the state must of the Supreme Court to prove that the United States is committed to the rights that the country has no rights in the right.
At the same time of the court, the Court of federal government, in which the U.S. government
```
[128 tokens, no EOS]

### agreement_gap

prompt: 'The students who had spent the entire semester preparing for the final examination in organic chemistry'  [16 tokens, starts scrolling out at generated token 49, fully gone by 64]

draw 1:

```
The students who had spent the entire semester preparing for the final examination in organic chemistry and the study and development of the project. This essay is in the test.
```
[stopped at EOS after 16 of 128 tokens -- the model ended the document]

draw 2:

```
The students who had spent the entire semester preparing for the final examination in organic chemistry, as far as a result of the first time to support the best possible of the problem, is to be able to use more sophisticated elements to make a successful understanding of how possible food and water quality can be beneficial for plants.
As with these are different sources and they are not a good idea to make them feel of that waste and the natural gas. Some applications include the best of the food for these nutrients.
What is the need for a high-temperature diet for a home and high-quality diet that has been done by the local community of people. A good example, when this happens when the soil continues to grow
```
[128 tokens, no EOS]

### attribution

prompt: 'According to a study published in'  [6 tokens, starts scrolling out at generated token 59, fully gone by 64]

draw 1:

```
According to a study published in the Journal of Child (OBA) and the National Cancer Institute, it was introduced to promote health and safety.
The study found that an area of this study was a growing process in the development of the American Medical Association, at the University of Chicago, the American Medical School of Medicine, and the National Medical Center for Disease and Prevention, and at the University of Applied Science and Sciences, and the University of California. Our study found that both in most cases of individuals who have diabetes or HIV are among many of the most likely to be infected.
There are several of these types of diseases in general, including:
- The
```
[128 tokens, no EOS]

draw 2:

```
According to a study published in the National Institute of Medicine, researchers who noted the study at the University of Michigan Medical Association.
```
[stopped at EOS after 19 of 128 tokens -- the model ended the document]

### dialogue

prompt: '"I do not think that is correct," she said, "because'  [14 tokens, starts scrolling out at generated token 51, fully gone by 64]

draw 1:

```
"I do not think that is correct," she said, "because I was going on a school and I’m just going to think, 'I am not a history teacher. I’d like to be my favorite, but I’m doing a teacher, but I’m not a way to teach her at the end.
My boys are a teacher, but I’ll be a teacher that is a great way and a son who doesn’t need, but I really have a school. But I’m sure I think there’s all that what you’re a one-year-old parent. I’m very excited
```
[128 tokens, no EOS]

draw 2:

```
"I do not think that is correct," she said, "because, in the right, I’ll like to be my favorite."
And the first step to the post is the same. But, I’ve never heard of in my favorite books by the two boys, and I’m here. She’s a great deal than not just my own, I would like to have a 5nd and would never be able to move to this new class. Thank you!
My kids are 10. I’m no, and I’m love the American History Math books. My son is the most amazing!!
My daughter is 11, 9, 6
```
[128 tokens, no EOS]

### factual

prompt: 'The capital of France is'  [5 tokens, starts scrolling out at generated token 60, fully gone by 64]

draw 1:

```
The capital of France is, one of the other countries on the United Nations. A few of both the world’s most impressive business and the first economic growth. Some of the most important things have been made to be done and will be a great challenge for the nation to be.
The European Union in the region of America, however, is a city of Japan. Its name is a city of a country. An industrial history of China is created by the world’s largest island.
The European Union is the city’s oldest largest and regional population, the city of East Africa, and the city of Mexico.
- The city
```
[128 tokens, no EOS]

draw 2:

```
The capital of France is far beyond a high price tax for an hour (p = 8, for example, the average budget of capital, which is the federal number of a number of votes; it is the value of a total tax income, and the average income is that the two are equal to the income.
The average of total income in a total of $6,000, while the average tax between the income of goods is 10,000.
The average current income of tax debt in the total number of tax assets tax is 8,2 U/2 years.
The tax score is the percentage of £1,500.1 billion of
```
[128 tokens, no EOS]

### numeric_units

prompt: 'The mountain rises to a height of'  [7 tokens, starts scrolling out at generated token 58, fully gone by 64]

draw 1:

```
The mountain rises to a height of 5%. The average rainfall is 6.3 days and the average distance of the population is 6.8 kWh (7.1 inches) and in diameter. The average annual rainfall.
The estimated increase in the amount of land was estimated at 1.9 cm.
- If the winter is high, it will not be more important of the size of the soil.
- The average annual tax between plants in the atmosphere is 10.1% in the growing cold water (4.7 kg) in the water by 10.25 mm (6.2 kg pere) = 0.10 m and 1.4 kg
```
[128 tokens, no EOS]

draw 2:

```
The mountain rises to a height of 25 feet (8 mm above) and a half on the side of the river, in the north, and the south. The mountain is the river is of the south. When it flows a mountain of the river, the river is the river and is its city, it is a man which appears to be part of the forest and the mountains, the Great Sea of South America.
The Great Plains in the South West, South and northern parts of North America have been in the United States. Although the Mediterranean cities are in the urban market, it shows the most important in the United States.
The New Brunswick-funded areas of
```
[128 tokens, no EOS]

### code_ood

prompt: 'def fibonacci(n):'  [7 tokens, starts scrolling out at generated token 58, fully gone by 64]

draw 1:

```
def fibonacci(n):5–20.
- “The most commonly used ‘turin’ to the same type-time-size-term-flowing-back-to-hand-step’ (1) vp (i) for the ‘C‘gin-fiberion-led,’ (e.g., ‘M-1)’ (c. to ‘d“m’), a “dong” (p.m., m., “gis”, “Q”) “macity”
```
[128 tokens, no EOS]

draw 2:

```
def fibonacci(n):5-2-8.
- Chavaske K. (2008). "This is one of the seven reasons of survival in other species, at the same time, and so on the other, the extent that the fossils are very distinct from the same species. The most common dinosaurs, are found in the wild and southern areas of the Mediterranean region, the northern and midचलर डिपरियासागाक नसिंम्गामवःस�
```
[128 tokens, no EOS]
