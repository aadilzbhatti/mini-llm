# Sample report

- checkpoint: checkpoints/ckpt_blk128_emb256_head4_layer4_bs64_steps15000_lr0.0012_minlr2e-06_seed42.pt
- step: 15000
- params: 16,105,297
- config: {'vocab_size': 50257, 'block_size': 128, 'n_embd': 256, 'n_head': 4, 'n_layer': 4, 'dropout': 0.0}
- eval_train_loss: 4.292113435268402
- eval_val_loss: 4.51638423204422
- full_val_loss: 4.541521743162106
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
Photosynthesis is a process that encourages the creation of the new world of plants that can be found in the early part of their growing world. The term ‘The Earth’ is a method of making new plants and plants. They are a part of the study of how a plant’s life and what it is called the ‘brain’.
One of the most important things is the Earth. With all the essential nutrients, minerals, and nutrients that are produced at the center of human growth and growth of the planet, it is important to examine and understand the basic concepts that will help keep the growth, development and growth.
A well-designed biofeeding system offers numerous benefits for plants and plants, including plants, plants, and plants, for food, and for pets.
Culture, gardening, and soil management are a fantastic way to keep plants in mind:
- Plants need nutrients to be a good source of health and wellness, in order to meet the needs and needs of plant.
- Trees need nutrients to be in the process of making your plants healthy and healthy.
- When mulching in the soil, the soil will absorb food that contains the nutrients, proteins, and nutrients that are essential for the soil.
- Vegetables can help boost
```
[256 tokens, no EOS]

draw 2:

```
Photosynthesis is a process that requires a chemical reaction in time in the cells cell, which is very easily transformed into cell-sensory cells (or mitochondria).
This method is used in many things, including the cell cell line. This method is the method of using a combination of cells. This method is commonly used to measure the energy consumption, the energy consumption, and the energy consumption. This technique is called the additive or additive (a process) and is used to determine the amount of light to the cell to be measured according to the number of factors. The synthesis of these two factors has been used for each generation of cells, and the amount of light is converted by the input to a single cell.
The method of transformation of the cells is used to calculate the value of the cell in a cell line. It is also used to determine the value of the cell line in a cell line. The method of transformation of cells in a cell line can be used in many different applications.
The method of transformation of cells is used to generate an electrical output. This method can be applied to the cell, since the process process is replaced by a cell line. This is called the transformation of cells. The change in the cell line is then added to the cell line.
The process
```
[256 tokens, no EOS]

### biography

prompt: 'Albert Einstein was a German-born theoretical physicist who'  [10 tokens, starts scrolling out at generated token 119, fully gone by 128]

draw 1:

```
Albert Einstein was a German-born theoretical physicist who took the first and next two years of mathematics. He is a scientist who takes part in his life’s career as an inventor, or a lecturer in mathematics and physics. In 1869, Dr. John Louis was one of the first young mathematician mathematicians.
This is a great part of the study of mathematics in a small geographical area of the Earth. It is a big difference in the geometry of the Earth’s surface and that the gravitational characteristics of the planet were quite very short and the ones where they were most commonly found.
The study of the Earth was published in The Earth’s first column.
The survey of the sun was published in the journal Human Anatomy of the Sun.
In 1891, in the journal Nature Biology, the scientists studied the planets in the Solar System of the Solar System.
According to the study of the Earth, the solar system is represented by the atmosphere and the surface of the sun.
According to the study, the sun's surface is still cloudy, and the atmosphere continues to warm. The moon is the sun and is the Sun.
The planet is the sun where it is going to rise and it is located.
This planet is the planet orbiting the Earth.
Its
```
[256 tokens, no EOS]

draw 2:

```
Albert Einstein was a German-born theoretical physicist who studied the theory of the universe. Aristotle’s discovery of a new theory of relativity can be an important aspect of life, such as in the brain, which is how humans are involved.
The other part of the theory of physics is that human existence can occur within the sphere; in the other hand, all of which have an eternal existence of space, can be achieved, or the physical being, a matter of a particular object. However, in the first few years, the theory of relativity is not very new, but we do not have any limitations.
How is Newton important to the theory of physics?
Mag Newton has been known to have no specific and important theoretical knowledge to prove. The theory of relativity shows that the theory of relativity has an influence on the mechanics of gravity. The theory of relativity is that the fundamental theory of relativity requires a process in which a number of objects are formed. The theory of gravity is that it is not a concept in the theory of relativity.
Mag Newton is a physical science that is based on Newton' theory of relativity, and does the theory of relativity. Therefore, Newton's theory of gravity is not much limited to the theory that Newton and Newton are trying to explain for magnet relativity.
He says that the
```
[256 tokens, no EOS]

### science_explainer

prompt: 'Oxygen is a chemical element with'  [8 tokens, starts scrolling out at generated token 121, fully gone by 128]

draw 1:

```
Oxygen is a chemical element with a highly alkaline state. It is then called the most effective and effective, but this is due to a new chemistry of the solution which is the only source of many enzymes.
As we mentioned in the “Lectin” section of the body, we just have a great deal of the chemical element. In that we have been given out the “liet” that we are going back to the beginning of the “Lectin” – the “rut” and the “rut” – the “rut” (i) that will be the most active of the cell – and which becomes the dominant element, as well as that they have come in with a chemical element – the most effective anti oxidant – they have been used for the body, because they have the same effect – they have no adverse effects.
In order to understand the difference between decay and disease – the presence of a plant – has a huge impact on this. The impact that we are making a difference is to be seen with the presence of a plant – if we look at it in my own way, we’re very knowledgeable about how to grow the plant from the root to the roots.
We
```
[256 tokens, no EOS]

draw 2:

```
Oxygen is a chemical element with high levels of potassium as a natural contributor to the antioxidant level of potassium, which is a complex element, especially in the coronary arteries.
However, this element of the diet is essential for heart health. It is known that certain dietary oils may be used in the body and are considered to be used in the body for high concentration of potassium. You can also use the following techniques in a study by the Food & Drug Administration (MDA) in patients with elevated sodium levels of folate and can be used to treat osteoarthritis.
In addition to the dietary habits listed below, the dietary values associated with vitamin C are higher among those who can help a person with calcium. By the above mentioned diet, it is important to recognize the dietary values that are essential for maintaining healthy balance. To prevent obesity, you should consult a dietician to identify the optimal levels of zinc supplementation that may be useful for your health and you might otherwise be more familiar with other conditions.
If you are considering your nutritional status, consider focusing on its nutritional status at a diet rich in vitamin C, you may have an option to eat. To avoid this, you should seek veterinary assistance and consult a healthcare provider for a vitamin C supplement.
- You can make a salad or a
```
[256 tokens, no EOS]

### instructional

prompt: 'In this lesson, students will learn how to'  [9 tokens, starts scrolling out at generated token 120, fully gone by 128]

draw 1:

```
In this lesson, students will learn how to solve problems.
- Students will learn how to solve this problem in addition to their children. The students will learn how to solve problems. We help students learn how to solve problems in their homeschooling and learn how to solve problems.
- Students will learn how to solve problems in the early stages of these problems.
- Students will learn the process of developing a problem in the first grade of the student
- Students will help them solve problems and problems with problems
- Students will
- students solve problems, which they know
- the problem solving and problem solving problem solving skills
- Students will learn
- they may also be involved in the
- The problem solving
- Students will learn and solve problems
If it is not
-
- students will learn how to correct problems
- how it works
- the problem solving problem solve &
-
- how to fix problems
- how to quickly solve problems
- how to solve problems
- how to solve problems
- how to solve problems
- How to solve problems
How to solve problems
how did a problem solve end
how does an problem solve problem solve problem solving problem solving problem solve problem solve problem solving problem solve problem solving problem solve problem solving solved problem solving problem
```
[256 tokens, no EOS]

draw 2:

```
In this lesson, students will learn how to read aloud. You will be writing a story. You will learn from the text by having read the texts and discuss the letter (or letter to the letter or letter) in the classroom or on the internet, the text is more than read. There are, however, some of the words that are written in the text are added to this book. It is not the same – although, it is easy to write in the middle.
My primary definition is your name, but you can read this book by writing a different text. My principal can also be read an article to the reader. Once you have read, click the word to indicate the meaning and composition of the whole text according to the text. It is an example of a “solar” as it is a bit confusing. The idea is a complex that the first verse of the sentence. The first verse should be clear: The second verse is the main theme of the passage, which is a verse in the central theme. The second verse represents the first verse of a verse, which is a verse from which the second verse gives the title of “solar.” – is an example in the book “solar”, which is the second in a verse, a verse
```
[256 tokens, no EOS]

### bullet_list

prompt: 'There are several benefits to regular exercise:\n- '  [11 tokens, starts scrolling out at generated token 118, fully gone by 128]

draw 1:

```
There are several benefits to regular exercise:
- ichthyosis
- ichthyosis and rheumatic obstruction
- ichthyosis
- ichthyosis (hypertherapy)
- ichthyosis
- ichthyosis or spasmic lesions
- ichthyosis
- ichthyosis (stymomous hypothermia)
- ichthyosis
- ichthyosis or ichthyosis
- ichthyosis and ichthyosis
- ichthyosis and ichthyosis
- ichthyosis
- ichthyosis symptoms
- ichthyosis
- ichthyosis
- ichthyosis
```
[stopped at EOS after 139 of 256 tokens -- the model ended the document]

draw 2:

```
There are several benefits to regular exercise:
- erythrocystic: it is not recommended to go over the counter mind. It is recommended to avoid excessive exercise or reduce the risk of the condition.
- erythrocystic: it is possible to prevent a worsening of a fever, which is also recommended to prevent further complications from getting back to normal.
- erythrocystic: It has a very short-term effect on certain types of headaches. It is not recommended to prevent any symptoms of a sore throat or a sore throat or a cold or sore tongue or face.
- erythrocystic: It is recommended to use a balanced diet or a diet-based diet. It is also recommended to use a balanced diet to avoid a sore throat.
- erythrocystic: This vitamin is often considered the first step towards the absorption of cholesterol, and it is also used to treat the condition. This may be done with regular exercise or lifestyle changes.
- erythrocystic: It is important to know if you think you are losing weight.
- erythrocystic: Pain in the neck, stiff neck, or a tight neck, the affected bones and the bones with a normal balance of texture.
- erythrocystic: This
```
[256 tokens, no EOS]

### numbered_list

prompt: 'To solve a quadratic equation, follow these steps:\n1.'  [15 tokens, starts scrolling out at generated token 114, fully gone by 128]

draw 1:

```
To solve a quadratic equation, follow these steps:
1. When you add of a quadratic equation,
1. If you're using the Pythagorean Formula,
2. If the formula is wrong then the unit is left on the sphere of the equation.
2. If you're using a quadratic equation or the formula, you can use a quadratic equation with the right formula.
4. If the quadratic equation is left on the unit of the quadratic equation, you should use the quadratic equation of the coordinates of the quadratic equation.
5. If the quadratic equation is first, you should get the quadratic equation:
The quadratic equation is given to convert the quadratic equation of the quadratic equation (the quadratic formula) as the quadratic equation is to convert and subtract.
This equation is named after the quadratic equation. It is called the quadratic equation and is a quadratic equation. The quadratic equation is called quadratic equation, called equation - "(a + ) \t^q)
The quadratic equation is given as the quadratic equation of the quadratic equation, the quadratic equation, and the quadr
```
[256 tokens, no EOS]

draw 2:

```
To solve a quadratic equation, follow these steps:
1. In case you are using a quadratic equation and q =, the quadratic equation for the quadratic equation is:
1. For the quadratic equation, you are using a quadratic equation.
2. For the quadratic equation, you can set up the quadratic equation.
a quadratic equation for the quadratic equation.
a quadratic equation. To identify quadratic equation, the quadratic equation is a quadratic equation with a quadratic equation.
a quadratic equation of dividing the quadratic equation,
b quadratic equation, and one quadratic equation.
b quadratic equation.
b quadratic equation is the sum of three quadratic equation.
b quadratic equation formula
c quadratic equation is the sum of the quadratic equation.
a quadratic equation is the sum of the quadratic equation.
b quadratic equation of the quadratic equation
b quadratic equation is the sum of the quadratic equation.
c quadratic equation: the quadratic equation is the quadratic equation.
c quadratic equation is the
```
[256 tokens, no EOS]

### enumeration

prompt: 'There are three main types of'  [6 tokens, starts scrolling out at generated token 123, fully gone by 128]

draw 1:

```
There are three main types of a good cholesterol cholesterol, which is the most common type in the body.
A good cholesterol is a good cholesterol, which is a good cholesterol (the high cholesterol cholesterol). It helps to reduce blood cholesterol and the risk of heart disease, stroke and other diseases.
Protein has an important role in the risk of heart disease, stroke, diabetes, cancer, and atherosclerosis. It also helps to manage cholesterol and cholesterol.
In some women, heart disease is one of the most common causes of heart disease. Although it can be a natural way of life-longed way, it can cause heart disease by gradually increasing the risk of heart disease and stroke.
A diabetes treatment of diabetes is a major problem for people who are overweight, according to the Centers for Disease Control and Prevention, stroke screenings are a major problem with the disease.
There are various factors in determining whether a person’s medical professional needs a high blood cholesterol level, and that could help you get your high blood cholesterol levels.
What are the side effects of hypertension?
The main cause of hypertension is the fact that your risk will be higher in height, but it’s important to remember that a high blood cholesterol level is high. However, while cholesterol levels are
```
[256 tokens, no EOS]

draw 2:

```
There are three main types of diabetes among people.
- Vitamin D, which can also cause heart disorders, such as Type 2 diabetes, diabetes, or chronic obstructive diabetes.
- Vitamin B12 is another type of diabetes that causes stroke in people with type 2 diabetes.
- Vitamin A is the major risk of type 2 diabetes and heart diseases.
- Vitamin B5: Vitamin B12 is a type of diabetes that affects people with type 2 diabetes, but this type of diabetes can become a major disease affecting people with type 2 diabetes.
- Vitamin A
- Vitamin K: While most people have diabetes, they can be hard to eat.
- Vitamin D is linked to cholesterol, which is a common problem of a person with the type 2 diabetes.
- Vitamin D: This is the leading cause of heart disease.
- Vitamin C: This causes a buildup of the liver of the body.
- Vitamin D: This can be hard to eat without it.
- Vitamin B is the most important and essential vitamin to eat them.
- Vitamin B is required for vitamin D levels.
- Vitamin B is the main cause of the heart.
Some foods that take vitamin to include vitamin B12, vitamin B12, and vitamin D to protect against certain types
```
[256 tokens, no EOS]

### long_dependency

prompt: 'Although the treaty was signed in 1919, it'  [9 tokens, starts scrolling out at generated token 120, fully gone by 128]

draw 1:

```
Although the treaty was signed in 1919, it was the first British-leaning War between the United States, which opened its first draft by the British government, and it has to be signed in the House of Lords.
As a result the United States imposed a Union treaty on Palestine, which was to be signed by the Continental Congress on a day of Palestine in which the United States ended up with the U.S. Constitution.
By the end of the United States, Israel did not stop the British treaty until then.
The United States was formed in 1939. While Congress began to enact the first constitution in 1938, Palestine entered and the Second World War would have been the only time to enact the treaty.
On the first day of the treaty, Jews returned to their land at the time of the Great War and to surrender their colonies. They were not allowed to surrender and surrender, and the Israelites were too poor, especially for their return, as the United States had failed.
In April 1859, the United States returned to the Senate.
Brunery refused. He had the right to resignorate after the United States. The General Assembly had planned that he and he did not meet the States as a nation, as a whole. He then moved to the United States. His parents were still
```
[256 tokens, no EOS]

draw 2:

```
Although the treaty was signed in 1919, it was the first Congress to set the treaty to the United States.
During the First Act, President Martin Luther King was a member of the United States in 1954 and served as the official official member of the United States for the United States.
Prior to the declaration of the President’s resolution, an American politician in the U.S. Navy was given a request to establish a legal and impartial decision on the status of the United States.
In February of 1946, President Lincoln signed its resolution on the federal basis of the United States, but resigned to take its place in place of the National Security Administration, the National Security Commission, and the National Security Commission, in which President George Clinton, President, President, and President of the United States.
The United States authorized the U.S. Military Defense Administration to ensure a secure and security-related protection against the U.S. war-related problems that led to the development of a new nation.
The United States government to build a comprehensive security system, a federal government-funded organization, and the U.S.-sponsored government-sponsored resolution between the two States and the U.S. U.S. and the federal government to foster a global security organization.
In conclusion, when a federal
```
[256 tokens, no EOS]

### agreement_gap

prompt: 'The students who had spent the entire semester preparing for the final examination in organic chemistry'  [16 tokens, starts scrolling out at generated token 113, fully gone by 128]

draw 1:

```
The students who had spent the entire semester preparing for the final examination in organic chemistry should be able to take advantage of the other students in a particular way. This is the result of the study of organic chemistry, which has been identified as the basis of the scientific research, and the development of the study is important.
A recent study of nutrition in the study found that there has been a number of studies that showed that the benefits of dietary supplements are high in that of the total available dietary supplement in organic chemistry, especially as of the majority of other foods that were not high in mineral magnesium. However, it can sometimes be said that by the study I mentioned above, other studies are important to the study of the dietary supplement found in each sample with no added dietary supplement, including:
- B vitamins, minerals and proteins - iron, phosphorus, calcium, phosphorus and minerals -
- B vitamins and minerals -
- B vitamins, minerals, minerals, and vitamins
- B vitamins and minerals -
- B vitamins A and E vitamins A and O are available to you.
- B vitamins A and E vitamins A, B and E, and E, are essential. Vitamin A and E, are essential to your body and kidneys. These are essential to your body.
C and B vitamins A
- Vitamin A
- Vitamin B6
```
[256 tokens, no EOS]

draw 2:

```
The students who had spent the entire semester preparing for the final examination in organic chemistry have been an excellent solution to the development of the curriculum. The student also experienced a grade of the students who had been working on designing new chemistry. All students who had a high grade of chemistry did not have the opportunity to be successful over the years.
As the student stressed, she had the opportunity to build the course of chemistry and chemistry. She continued to be able to demonstrate the experience with science. From there he had hoped the students could not be able to get the most out of their science and to make his own. But she was not involved the learning process to design the science of chemistry.
This lecture explains how this lecture is about how our chemistry was created and created, what the experiments it found.
What the chemistry has been around for my students?
A decade of research in chemistry was not an answer, but rather a problem, there could be an understanding of a study of the subjects and how they thought that they were able to understand.
```
[stopped at EOS after 195 of 256 tokens -- the model ended the document]

### attribution

prompt: 'According to a study published in'  [6 tokens, starts scrolling out at generated token 123, fully gone by 128]

draw 1:

```
According to a study published in London, the study of the work was published in 1886. The journal The Journal of the Journal of the American Medical Sciences reveals that this research is not a substitute.
The Journal of Physics is based on the study.
The research was carried out in the journal “The Role of Metresizing Your Hearing” by Dr. Carl Jung (2013) examined the effects of a hearing loss on hearing loss on hearing loss of hearing loss on hearing loss, and the effects of hearing loss and hearing loss as a result of hearing loss on hearing loss and hearing loss.
- The results, conclusions and conclusions regarding hearing loss and hearing loss are displayed.
- The findings of hearing loss can vary depending on the age of an hearing loss.
- The findings and conclusions of hearing loss may vary in proportion to hearing loss, and the data can vary depending on the severity of hearing loss.
- The findings suggest that hearing loss is a contributing factor.
- The results of hearing loss may be due to the genetic limitation of hearing loss due to hearing loss and hearing loss.
The findings suggest that hearing loss may be due to hearing loss, but the results are due to their increased hearing loss or sensitivity to hearing loss.
- The findings suggest hearing loss or
```
[256 tokens, no EOS]

draw 2:

```
According to a study published in Nature Biology, the researchers have now uncovered that the brain can interact with the ability of the brain to predict.
Researchers now have developed a new approach for detecting certain brain responses that will provide insights into the way the brain responds to change. This would be a promising way to identify the underlying causes of brain development.
Researchers have looked at these changes:
"These changes may have a significant impact on the brain, and it is more likely that they do not have any effect. They will try such action on how the body responds to this kind of neuropath.
"This may have implications for how neuropaths interact with the brain," said the study. "If the person thinks that they're feeling more like that, or the other."
Dr. Robson said, "If you're doing something that you're doing, you're doing."
Prof. Seiurea told us that in the next few years there is a good chance to get a "I" or "I'm doing."
"Let's say you's gonna change the effect."
In the meantime, there's no way to make sure you're so much so much more. You'll get the help for you to help, however, by finding out more about what you
```
[256 tokens, no EOS]

### dialogue

prompt: '"I do not think that is correct," she said, "because'  [14 tokens, starts scrolling out at generated token 115, fully gone by 128]

draw 1:

```
"I do not think that is correct," she said, "because we don't know how to use this word."
I'm talking to the doctor, and my health, of course, "I'm really telling you the truth you've heard."
HIV's "I'm going to talk about a way of understanding the ways of learning."
HIV can be used in any way by reading the words, and it's just a simple way to talk about things like a language that is actually "I'm going wrong" with something.
"I are going to be "I'm going to be "I'm not going wrong." That's just a big piece of thought and that's the reason for a big topic."
HIV's "I'm going wrong." It's just a small piece of books.
She's talking about it. "I'm going to tell me how to answer?"
She's stuck up when I'm going to write his or her statement. "I'm going to go wrong."
She's talking about his "I'm going wrong."
She's been very sad, and I'm going to write his speech to make it.
"I'm going wrong," says Kiela, "I'm going wrong."
The fact that I'm going wrong is that I
```
[256 tokens, no EOS]

draw 2:

```
"I do not think that is correct," she said, "because it will not be the case," he said. "If there is a problem, it appears to me, "
There's a problem that's not the case," she said. "That's why you'd like to think you'd not really have some sort of thinking about it."
She said, "I'd like, "I'm really really interested in the way is I'm the "self", "I'm," "but."
Well, "I'm not."
"I'm up- of a "hmmm."
"I really want "to give an opportunity to think, but I'm probably really like."
"I've got to "mixed" the "very-p" button."
'I'm not afraid of my friend's name "I'd like it!" "I'm not happy."
"I'm going to say "I'm not really excited."
"I'm going on to say "I'm going to go."
"I've got a word in "The "I'm going."
"I'm not going to go to say "The "I'm going to you," and I get the "What's going"".
"My little is my friends," said the "My
```
[256 tokens, no EOS]

### factual

prompt: 'The capital of France is'  [5 tokens, starts scrolling out at generated token 124, fully gone by 128]

draw 1:

```
The capital of France is the capital of London and Ireland. The capital of the Netherlands is known as the capital of the French.
The capital of the French settlement is the capital of the Kingdom
in India and Ireland in the region of the country.
The capital of the Kingdom was the capital of the Kingdom and the region of Spain, with the capital of the Kingdom of England.
Spain was the capital of the Netherlands.
Spain was the capital of the Union of the Republic of France, and the first national capital of the Netherlands.
Spain was the largest city in the province of the Netherlands.
Spain is the capital of the British and Dutch.
Spain is the capital of the Italian capital of the Netherlands.
Spain is one of Spain's capital of France.
Spain is the principal city capital of the world, and country is the capital of the country.
Spain is a world of foreign countries, with its capital with a capital of 4,000.
Spain is one of the major regional states in the state.
Spain is the fifth largest country in Latin continent, the country of the city of the country. Spain is a country that is now 1,800 km². Brazil is the largest country in Asia.
Spain is the fifth largest country in Asia
```
[256 tokens, no EOS]

draw 2:

```
The capital of France is an important part of the world. In the first place, it is the capital of the country to do with the capital of France.
In 1871, the United Kingdom was the first of the United States to begin the Second World War of 1868, to begin and become a part of the Soviet Union in 1836, when the country gained fame. In 1866, the capital of Italy became the capital of Spain and was a city. During the First World War, France was the capital of the Republic of Italy.
In 1856 the Kingdom of Sicily was established. Belgium was the capital city of Netherlands and the Republic of Italy. It was first formed in 1775. It was the capital of Greece.
As part of the European Empire was renamed by Spain, the French had a square base and the Roman province. It was renamed for the Spanish government.
In 1869 the state was restored to Spain.
The French was built for
the French to be used. With the English
language of the French
- French of the Netherlands, and the Italian
- French
The Netherlands was a significant British and
- France, including the French.
- French was considered to be a major
- Portugal trade was the first African-American
```
[256 tokens, no EOS]

### numeric_units

prompt: 'The mountain rises to a height of'  [7 tokens, starts scrolling out at generated token 122, fully gone by 128]

draw 1:

```
The mountain rises to a height of 0.5 km²; the mountain flows below the equator. The mountain ranges from 5 meters to the nearest area of the city is 6.6 km.
An in the estuaries of the estuaries by The mountain ranges have been in the estuaries of the Aluc and the Mughal valleys. The mountain ranges range from 8 to 7.9 km from the estuaries.
The mountain ranges from about 7 to 10km (16.8 mi) to from the west to the north of the mountains of the mountains. When the mountain peaks are around 7,000 meters, the peaks at about 1,200 m (0.0 mi) will be about 1,200 m (0.5 mi) and the peaks at about 1.5 meters above sea are also about 1,200 m (1,125 ft). The mountains of the Sughal mountains are about 1,200 m (0.9 mi) so the mountains in Baja or Tjarar.
The mountain ranges are about about 3,700 km (30 km), the mountains of Nakhs. The mountain ranges are about 2,830 m (1,200 m) and 12 km (1,800 m) and 8 km
```
[256 tokens, no EOS]

draw 2:

```
The mountain rises to a height of 5 feet, so not a full moon, is a high-lighter and thick-shaped area.
The mountain ranges, which are low, are not strong, but are low, but in the low, and above sea level, the state is divided by the upper, lowest, and lower, and lower, and are less likely to be overweight-bearing, and thus larger.
The mountain ranges offer a variety of snowfall, which also includes the mountain ranges, particularly the high mountains.
By the day of autumn, the mountain ranges in the state, and the mountains are similar to the mountain ranges.
The mountain ranges also have a unique mountain ranges. It also contains the mountains, streams and rivers, mountains, and deserts. This mountain ranges is a mountain range.
What is the difference between mountain ranges and mountain ranges?
Sea range is low. It is a mountain ranges of mountain ranges, high mountain range.
The mountain ranges are rich in sandy areas, and rivers, and can also serve as a mountain ranges.
The mountain ranges include the mountain ranges or coastal ranges, along with the mountain ranges.
The mountain ranges include latitude, slope, and the slope of the mountain ranges.
Are you the mountain ranges in latitude
```
[256 tokens, no EOS]

### code_ood

prompt: 'def fibonacci(n):'  [7 tokens, starts scrolling out at generated token 122, fully gone by 128]

draw 1:

```
def fibonacci(n):
- I'm a student,
- I'm a student,
- I'm your student,
- I'm a kid,
- I'm a teacher
- I'm a teacher,
- I'm a teacher,
- I'm a teacher and I'd a student.
- I'm a parent,
- I'd like a student or a teacher.
- I'm trying to make a classroom.
- I would like to learn how they can be able to give it a real time,
- I’ve thought about my classroom,
- I'm so excited about
- my teacher and my teacher might try to read or write on the board.
I would be able to give me an added bonus.
- My teacher helped me or her teachers and
- I think it really is my teacher
- I would love math
- I would get some simple ways
- I would do better
- I would like to thank those who did
- I would love math science but
- I would like to thank my teacher
- I would be interested in
- my teacher,
- I’m going to tell you that I could love math
- I would love math,
-
```
[256 tokens, no EOS]

draw 2:

```
def fibonacci(n): eukary(n): eukary(n)e (t)n)) c/est (re) n: eukary(n)e (n)e (n)e (n)e (n)t - (n)e (n)te (n)a (n)n)e (n)e (n), n)e (n)c (n)e (n)e (n)e (n)e (n)e (n)e (n)b (n)e (n).
n (n)e (n)e (n)g (n)e (n)e (n)e (n)e (n)e (n)e (n)e (n)e (n)e (n)e (n)e (n)e (n)e (n)e (n)e (n)e (n)e (n)e (n)e (n)e (n)e (n)e (n)e (n)e (n)e (n)e (n)e (n)e (n)e (n)e (n)e (
```
[256 tokens, no EOS]
