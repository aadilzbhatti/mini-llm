# Sample report

- checkpoint: checkpoints/data10k_160k_tied_bias_seed42.pt
- step: 160000
- params: 7,283,153
- config: {'vocab_size': 50257, 'block_size': 64, 'n_embd': 128, 'n_head': 4, 'n_layer': 4, 'dropout': 0.0}
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
Photosynthesis is a process that is highly efficient and can be used widely in the fields of chemistry for the development of hydroponic acid, but in the very mid-level of fossil-fueled carbon, and a low-cost alternative to the Western atmosphere as well as the basis of bio-scale bioto process.
- Biofuel revolutionized by the Energy Commission (CSF) with more than a data base of the International Energy Organization (WHO) that increases carbon emissions from foreign gas emissions.
- greenhouse gas supply chains.
- In addition to demand for renewable energy sources, carbon gain, and carbon emissions.
- to be more effective than
```
[128 tokens, no EOS]

draw 2:

```
Photosynthesis is a process that works for all, but it is less effective than a specific type of research used to meet the needs of our own studies.
The study is shown in the study of ASR gene, which is found in a number of species.
The study published a study of the study and clinical trials is using the two findings and the participants who are studying genes from the gene, from the researchers of a group of different types of phenotypic characteristics in the clinical trials. Additionally, the study identified that the genetic patterns of genetic theory have increased the success of the study of gene expression in the study and the effect of phenecology in the
```
[128 tokens, no EOS]

### biography

prompt: 'Albert Einstein was a German-born theoretical physicist who'  [10 tokens, starts scrolling out at generated token 55, fully gone by 64]

draw 1:

```
Albert Einstein was a German-born theoretical physicist who had ever been a small Americanist. He wanted to change their history, in the early 20th century, and the early African American American were not the subject of African Americans. It was a new language that she was the same in Europe - after the first twenty years, and was not a part of the history of the American American English.
The history of the year’s history began in a literary history of the world of the first century. She said that she also was a first in its history, but they did not have a very limited history. One of his great historical figures included on the American history and history of
```
[128 tokens, no EOS]

draw 2:

```
Albert Einstein was a German-born theoretical physicist who used a gun in the British lab. The only study was found that the first ‘a’ of the study was the first to be described (1).
The study was first published in the journal (2.1) in the American Journal of the American Medical Association).
“The findings were aimed to explain the effects of a number of participants, and the risk of developing the general population (2.1) in the group.”
“The study of childhood research was a study on childhood obesity and obesity.”
Another research paper found that the researchers are asking for the development of the latest
```
[128 tokens, no EOS]

### science_explainer

prompt: 'Oxygen is a chemical element with'  [8 tokens, starts scrolling out at generated token 57, fully gone by 64]

draw 1:

```
Oxygen is a chemical element with other properties in the chemical and chemical properties of a human body. Also, the reaction itself becomes less useful in reaction, the oxygen reaction is that the reaction is a hydrogen and the hydrogen ions form of which are determined to produce the process of a reaction. The equilibrium reaction is in the reaction to the reaction, which is the equilibrium of the radiation reaction, which is not yet only the equilibrium reaction. This increases ∠PO2 and other equilibrium, is the equilibrium of the equation of equilibrium. (HFC1.3)
The equilibrium of equilibrium will be equilibrium.
The equilibrium reaction equation of equilibrium
For only the equilibrium error
```
[128 tokens, no EOS]

draw 2:

```
Oxygen is a chemical element with a strong, and therefore, is extremely sensitive to other factors such as the primary cause, which can be used to describe the specific characteristics of the most important ones. This can be caused by the combination of different kinds of DNA species.
What is the genetic difference between species is the genetic problem that is not in the form of the DNA.
There are a number of genes that are found in the species of species. In the study of the species, the presence of the pUC18 gene and also the gene that has the gene from a variety of different forms. The gene content on the DNA and DNA from the human genome are
```
[128 tokens, no EOS]

### instructional

prompt: 'In this lesson, students will learn how to'  [9 tokens, starts scrolling out at generated token 56, fully gone by 64]

draw 1:

```
In this lesson, students will learn how to write ideas and learn vocabulary levels with all subjects. Students should be able to express any of these sentences to write the word so that they could not be able to read into their English.
What is the difference between grammar and grammar?
I believe that I believe that using a grammar class in my book, which is the first part of what is written.
My thoughts are different!
I think you are all trying to learn a language that is about the words you want to find. A person who wants to understand something they are written, and in some instances it will not be a great deal, it was still the least thing
```
[128 tokens, no EOS]

draw 2:

```
In this lesson, students will learn how to work together to read the role of teaching and writing writing. For this time, you can learn what will you learned for kids to work with reading and reading.
```
[stopped at EOS after 32 of 128 tokens -- the model ended the document]

### bullet_list

prompt: 'There are several benefits to regular exercise:\n- '  [11 tokens, starts scrolling out at generated token 54, fully gone by 64]

draw 1:

```
There are several benefits to regular exercise:
- 、 alternative heat-free water:
- Avoid excessive water: Regular cleaning, clean, or cleaning.
- Avoid following any heating methods: Flushing for a boil.
- Avoid burners, including moisture, water, and air, as well as water, water, and water drainage, and water.
- Avoiding drainage.
- Swelling for any kind of water, should be kept in the water.
- Avoiding water: When the water is cooled, water is absorbed by the water and its water.
If this process is completed, the water is installed (the oil water and minerals) and
```
[128 tokens, no EOS]

draw 2:

```
There are several benefits to regular exercise:
- ____________ - Why a person decides not to exercise?
Pregnancy is an important means to be a better job in sports and are. However, they are often at risk of stroke, which is still worth the risk of getting too much of sleep.
• Can You Eat Injuries?
Yes, if you buy a TV TV and a new game, itís a game that is a good way to take money to your school.
- How does you make reading about the game at a higher level.
- What is the difference between the game?
- How does the game originate from a game?
- How
```
[128 tokens, no EOS]

### numbered_list

prompt: 'To solve a quadratic equation, follow these steps:\n1.'  [15 tokens, starts scrolling out at generated token 50, fully gone by 64]

draw 1:

```
To solve a quadratic equation, follow these steps:
1.1. How can you calculate text?
1. What Is the Difference between PDFs?
3.4: What is it?
1. What is the difference between the text and the object?
2. What is a difference?
2 What is the difference between Word creation?
1. What is the main meaning of the text?
1. What kind of question is the function of a single-book?
1. Is the writer mind?
4. How to write an idea with our own title?
2. Do your name in your essay in a state?
1. What is your
```
[128 tokens, no EOS]

draw 2:

```
To solve a quadratic equation, follow these steps:
1.2.2.7.1/2
2.2.2.4 A quadratic angle of a circle.3.3
2.2.2.4
5.4.2.4 and 3.5.2
2.2.1.2.6.3
3.4.2
2.5.3.2.2.3.3.2.2.5.1.2.6.3.4.2.1.2.1.2.1.7.3.2.3.3.0.3.2.
```
[128 tokens, no EOS]

### enumeration

prompt: 'There are three main types of'  [6 tokens, starts scrolling out at generated token 59, fully gone by 64]

draw 1:

```
There are three main types of medical care options available for patient patients with patients. The treatment plan is to avoid that medical conditions like medical specialists.
What are the most common causes of the infection?
- Areontal infections a condition?
- Are causes of death or a cause?
These are the signs and symptoms that are present to the most common type of infection. This type of infection is usually a sign of the infection. You can avoid infection. It’s important to note that the infection does not cause you to come.
How many types of infection are caused by the disease?
How well do you know?
Dogs are
```
[128 tokens, no EOS]

draw 2:

```
There are three main types of medicines for a particular weight of the brain.
The main types of allergies include skin cells, skin, gastrointestinal tract, and digestive health.
- It is also the main form of the most common causes of an aging, which is the most commonly used in treating the disease, such as the disease, diabetes, and a long-term condition.
- This is the case of chronic pain in the developing arteries.
- While most of the most common symptoms of acne include:
- Certain symptoms that are triggered by the skin which is known as inflammation, especially in the mouth of the mouth.
- If you are infected and
```
[128 tokens, no EOS]

### long_dependency

prompt: 'Although the treaty was signed in 1919, it'  [9 tokens, starts scrolling out at generated token 56, fully gone by 64]

draw 1:

```
Although the treaty was signed in 1919, it meant that to reduce the death of the territory during the war. If he had died in the form of war between the 10th of Britain, the first and the British administration was passed on the border to the war, they needed to come up. (The Church of the 17th century, before the war, but on their own own, the Jews had to be more than the world, and that many tribes were given them from the end of their territory. The town's first-born, on the earth, and the other side of the whole country, and they had been in the city, as such as the United States for
```
[128 tokens, no EOS]

draw 2:

```
Although the treaty was signed in 1919, it was an extremely important part of the country's most closelyational and international civil engineering, in the early 20th century, even though the country had to increase awareness of the problem of history.
However, in the country, there were also a number of years ago in which the economy had a significant impact. Since the economy of the economy, the economy has become more sustainable and more environmentally-energy companies in the world. The economy had a rapid impact in the economy, and it changed the wealth of the economy.
In the era they grew at nearly once the start of the World Bank, we also believe in the world’
```
[128 tokens, no EOS]

### agreement_gap

prompt: 'The students who had spent the entire semester preparing for the final examination in organic chemistry'  [16 tokens, starts scrolling out at generated token 49, fully gone by 64]

draw 1:

```
The students who had spent the entire semester preparing for the final examination in organic chemistry can provide the opportunity to read our students' first, the students were able to learn more about the best.
1. What can we do to get to look at how to do?
2. How to Write a Project
If you are interested in finding a successful classroom, the student will need to take your little to use. We will need to use a more complex course of information. There is a lot of questions about the topic that will be easy to know.
Your students will need to be able to read about them. You have the opportunity to use English as you may be looking to the future.
My kids
```
[128 tokens, no EOS]

draw 2:

```
The students who had spent the entire semester preparing for the final examination in organic chemistry is still being able to perform the time they do much of this time.
I’d go into a deep soil to the other, but she’ll need to be the best part in this regard.
I’ve had a strong impact on soil temperature, but I’ve been seeing something on.
I’ve ever heard of the past. They are not just a bit of a matter of the surface of plant.
I’m going to see what we’re going to do, and then we will also get to give it a good food to learn about the problem.
```
[128 tokens, no EOS]

### attribution

prompt: 'According to a study published in'  [6 tokens, starts scrolling out at generated token 59, fully gone by 64]

draw 1:

```
According to a study published in March 20, the study of the National Institute of Agricultural Research and Innovation (CDC), which is a “low estimate of low-income areas” of the American diet.
With a global report from the Institute of Nutrition and Nutrition, the report concluded that over a month, the effect of food, the evidence suggests that women’s diets would increase the risk of developing and living. According to the study, the research findings have found that certain vaccines had a chance of receiving CO2.1 infection in the United States.
Many studies suggest that the population is no more than 022 years.
The survey found that
```
[128 tokens, no EOS]

draw 2:

```
According to a study published in New England and also published a report on the American Academy of Sciences.
It is worth noting that the question is that more than two people in the world are on the basis of the effect of childhood obesity in our life. The most common thing it can be used to predict the problem of the situation. With the help of the problem, an anxiety attack can cause significant negative effects on children’s wellbeing, more about what you are able to do for future generations.
The effects of dietary problems have been shown to lead to a lot of problems. For example, the patient is using a medical tool that uses vitamins and minerals.
```
[128 tokens, no EOS]

### dialogue

prompt: '"I do not think that is correct," she said, "because'  [14 tokens, starts scrolling out at generated token 51, fully gone by 64]

draw 1:

```
"I do not think that is correct," she said, "because of the world being able to live our planet, but also because of the truth, we believe us to our human world is a constant reminder of our nature. And what is we will, but we will tell us how a world could be. The first thing at any point of creation is that the planet is to say that the idea of the universe remains.
I’d like it is a way of how it’s the way we remember to see the universe you’d like to see this. I would like to see it, if there is a point to look at the Sun, and how we can you
```
[128 tokens, no EOS]

draw 2:

```
"I do not think that is correct," she said, "because the 'fimbrane', 'I" is not in all our eyes, and he is not.
How does Australia do our own health?
The American people, however, are not good for the best.
In other words, researchers will be in the case of the first and third to what is the law of food at home where they are in particular. The case of the US has been made by the United States.
The American Society of Engineers had to set the State Bureau of Energy and Development and the Council.
The American Wildlife Service (TIT) is an international agreement that is a record of the
```
[128 tokens, no EOS]

### factual

prompt: 'The capital of France is'  [5 tokens, starts scrolling out at generated token 60, fully gone by 64]

draw 1:

```
The capital of France is the first railway of the United States, with a number of total rail-Traction, the highest capital, with the same height to the capital of Belgium, according to the United States.
According to the federal district of Belgium, it is located in a province of Belgium, Belgium, Belgium and Belgium. The country in Belgium is the largest country in Belgium, Belgium, Belgium, Belgium, Belgium, Belgium, Belgium, Belgium, Belgium, Belgium, Belgium, Belgium, Belgium, Belgium, Belgium and Belgium.
The Belgium is the capital of Belgium in Belgium.
 Belgium divided by Belgium.
 Belgium.
 Belgium. Belgium.
```
[128 tokens, no EOS]

draw 2:

```
The capital of France is a very important factor to the economy. It is the largest part of the world.
The United States, which is the development of the economy and its economy, the economic crisis, and the market economy. In fact, the sector of the country has a relatively short time around the world. The economy has increased the growth of the world’s economy, and the economy is also important.
Fiberibility is the importance of the world and the economy and the economy and the economy has increased.
The economy is not the development of market development and has been built on its production of goods, goods and services, and goods
```
[128 tokens, no EOS]

### numeric_units

prompt: 'The mountain rises to a height of'  [7 tokens, starts scrolling out at generated token 58, fully gone by 64]

draw 1:

```
The mountain rises to a height of 2.5 feet away from the middle of the sea with a number of people. The area of which is covered with three other areas.
By the end, the trees are underused by an annual temperature.
Where are yellow red, yellow, red, orange, and reds.
The red has shown red-green shrubs and, which are commonly found in the red-yellow-green and yellow-brown (P4).
The red-brown leaves are similar to the “boll” flowers. They are found in the yellows and the growth of the yellow- purple plants that form the color
```
[128 tokens, no EOS]

draw 2:

```
The mountain rises to a height of 5.4°.
Culture and Environment
The Great Depression
The future and national importance of the growth of a population of 5.5-9 million
The World Climate Action
We will continue to be able to grow in the environment
a new population of 1.4 million years ago
The first major European countries that are around 50 percent of the population.
The World Policy and the European Congress’s Bank is the International Agency for Climate Change (WHO) that is a global security crisis that has contributed to our climate, as it remains, and it is not necessarily to be in the early years it is
```
[128 tokens, no EOS]

### code_ood

prompt: 'def fibonacci(n):'  [7 tokens, starts scrolling out at generated token 58, fully gone by 64]

draw 1:

```
def fibonacci(n):
Molei’s co-localization of CSPa and VIFR radiation with SOP, which is considered in this regard, in order to support the development of the product.
LIFS is a large number of layers called SOP in a way that is the first time, the COPs may be able to carry out an electrical system, as well as a high, an electrical engine that operates from a low level, which may be used to measure the flow of a wire. For instance, the other one can have the same circuit.
There are a different types of batteries that go from
```
[128 tokens, no EOS]

draw 2:

```
def fibonacci(n): (n)
(c) The equation
(c) The equation = 0.
(c) A = 1/4/e (3)
(3) =
(d) = 0.95 (x1) = 0.70
(d) = 1
(b) = 0.66 = (1)
(d) = 0.75
(x = + 2) + 2 + 1.60 = 1.
 = 0.99 × 1) =
d) = 0.80 x = 12;
b) = 5 = 0.75 × 1;

```
[128 tokens, no EOS]
