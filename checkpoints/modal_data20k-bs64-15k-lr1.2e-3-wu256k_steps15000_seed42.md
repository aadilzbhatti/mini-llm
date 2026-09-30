# Sample report

- checkpoint: checkpoints/ckpt_blk128_emb256_head4_layer4_bs64_steps15000_lr0.0012_minlr2e-06_seed42.pt
- step: 15000
- params: 16,105,297
- config: {'vocab_size': 50257, 'block_size': 128, 'n_embd': 256, 'n_head': 4, 'n_layer': 4, 'dropout': 0.0}
- eval_train_loss: 4.124248313903808
- eval_val_loss: 4.63265643119812
- full_val_loss: 4.656403407755545
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
Photosynthesis is a process that can be traced back to the ancient Greek and the ancient Greek to the ancient Roman Empire.
The term “canada” originated by the Romans and kingdoms of Spain, but the term “bape” was introduced to Roman a Roman rule. Here, the Romans were divided into two categories: the Romans, Roman Catholics, and Romans. Ancient Romans were usually called Roman or Roman. Roman Roman Catholics are usually dated by Roman Greek origin, most Christians, who settled in Roman or Roman Roman Roman. Italian English is a Roman state, or term derived from Roman or Roman. Roman Roman Roman Roman. Roman Roman Roman Roman Roman Roman. Roman Roman Roman is a Roman term. Greek is believed Greek. Roman Roman Roman Roman, Roman Roman Roman Roman, Roman Roman Empire, Roman, Roman, Roman, Roman, Roman and Roman periods. Roman Roman Roman: Roman, Roman or Roman Romans. Roman Roman. Roman Roman Roman: Roman in Roman. Roman Roman Period. Roman Roman or Roman Roman
 Roman Empire, Roman Greek in Roman also known Roman. Roman Roman Roman Roman for Roman Roman. Roman Roman in Roman was Roman. Roman Roman soldiers during Roman period Roman, Roman, Roman and Roman. Roman. Roman. Roman Roman, Roman. Roman. Roman.
```
[256 tokens, no EOS]

draw 2:

```
Photosynthesis is a process that requires understanding of the nature of biological processes and thereby, to perform and to optimize these processes. The introduction of chemical reactions in organic matter is critical to understanding the mechanisms that react together to complex conditions and to efficiently conduct biological processes. These factors include factors like oxidation number, oxidation number, decay number, and changes the energy balance of the organism. These processes, like nitrogen isotope, are called nitrogen in soil. The main differences in ammonium concentration and its processes are of complex and chemical reactions to carbon atoms.
- The chemical reactions of natural environment in organisms is therefore the most critical in the ecosystem functioning of the organism.
The structure of the organism is the basis for the determination of other organisms.
- The organisms that reside within the organism and each other is the nucleus of the organism.
- The biological environment is the presence of the organism and environment in which the organism is the basis of the organism in the organism.
- The organism has an important relationship in our organism (see:
- The organism is an adult organism;
- the organism;
- the organism with the organism, as the organism, the organism.
- In the organism, is the organism of the organism, and the organism.
- The organism undergoes the organism
```
[256 tokens, no EOS]

### biography

prompt: 'Albert Einstein was a German-born theoretical physicist who'  [10 tokens, starts scrolling out at generated token 119, fully gone by 128]

draw 1:

```
Albert Einstein was a German-born theoretical physicist who took the first and most of his colleagues at the University of Chicago, the first director of his research at the University of Cambridge in Technology, Cambridge, Cambridge.
The findings of his research in the research in the field of quantum mechanics were published in a journal of Theodor University.
A groundbreaking study of the human brain in the 1950s, published in the journal of the journal Proceedings of the American Academy of Sciences and Science at the Pennsylvania University of Pennsylvania, said that intelligence could have contributed to the understanding and implications of quantum physics.
The author added that an algorithm based on the study of the matter and the need to find a useful model for quantum theory in quantum physics.
However, the researchers found that neural networks are more widely used to detect chemical reactions in a wide range of fields.
They found that the particles of particles of superheavy particles and they found that particles of superheavy particles would be less than 0.7 and fission particles, but they also used to measure the distances of neutrinos.
"We have demonstrated that when the particles of supernova be smaller than particles that are at high, they can be charged with a very high degree of stability and particle size," he explains. "If the particles have an average velocity, the
```
[256 tokens, no EOS]

draw 2:

```
Albert Einstein was a German-born theoretical physicist who created the theory for the universe. Aristotle founded the theory of the universe itself.
The idea of an Earth is a whole of various disciplines in order to understand the universe, but the universe is a universe that is a universe that is a universe.
The theory of the universe is thought that the universe is thought to be the universe. But the universe is a universe that is not known and cannot be invented.
In theory, it is not the first way to find answers to Earth from the Sun and on earth.
The universe is a complex universe that is formed by the universe and its existence. The universe is a sphere of all of the universe and has a universe that is formed to be called a universe.
One of the first stars, called The Moon, is the first in the universe. It orbits the second closest and first, for the Greeks, is a star. A star is the third planet, by the Sun, and on the second planet.
A star is a star that is between Earth and the third planet. Two stars are composed of three types, ranging from the Sun and the third planet. The first stars consist of three layers and four orbitals. The second, the third planet, is the second planet, the second planet.
```
[256 tokens, no EOS]

### science_explainer

prompt: 'Oxygen is a chemical element with'  [8 tokens, starts scrolling out at generated token 121, fully gone by 128]

draw 1:

```
Oxygen is a chemical element with a high concentration of 15-hydroxyquinoline. Phosphate has been found to have been associated with a strong immune response that has been linked to an increased resistance to oxidative stress and environmental effects. In addition, these compounds are still used to combat the effects of these compounds in a wide range of chemical activities.
Possible effects of sulphate activity on the body
A common misconception that we are able to identify when to use these compounds is that they cannot survive the same chemical conditions in either one or two. These compounds can be found in many organic compounds, such as amyloid (such as nitrites) and for example, in the organic (organic) and organic (organic) and, inorganic compounds. In other ways, they are found that nitrate is naturally present in many organic products.
A lot of things that are considered to be useful in good health.
The key role of this herb is to be of different importance to those who form the herb in a variety of things. The other elements that we are organic is organic, organic and organic, which are known to be the same in organic matter.
It’s about the fact that these two factors can be considered organic.
The root of the herb inorganic organic
```
[256 tokens, no EOS]

draw 2:

```
Oxygen is a chemical element with high levels of soluble compounds. Therefore, both the soluble and solublines are adsorption, the oxidation of soluble compounds and their properties are the solublines compounds.
Bioengineering, Issue 72, Chemical, Molecular, Biochemistry, Carotenology, Neoproteric aromatic, supernator, bifacial, hydrophysol, hydrol, phenotypic, hydrol, potassium, lutein, sodium, potassium, potassium, magnesium, potassium, carbon, water, and vitamin A. The substances and the substances in solublines are both biological and biological and are therefore also linked to the regulation of the regulation of glycation.
Antioxidants are a source of anti-oxidant compounds. It is believed to be responsible for the formation of an organ, such as proteins and antioxidants, and can be made into the form of anti-oxidase and antioxidant materials. This means that it prevents absorption of the elements or elements of toxic gases, such as benzophosphory breaks, which are derived from polysaccharides, that are absorbed by bacteria.
In addition, the role of an ant can be toxic to the body, which can be found in various other skin tissues.

```
[256 tokens, no EOS]

### instructional

prompt: 'In this lesson, students will learn how to'  [9 tokens, starts scrolling out at generated token 120, fully gone by 128]

draw 1:

```
In this lesson, students will learn how to use a word or phrase to translate an appropriate word into this, and then use a word to communicate.
For this lesson, students will learn to use as a reminder of how to express them in sentences and using words from words to words. It can be very beneficial to create sentences by using a word to keep them in a word with words. This form of words with letters is meant to express a person or to use a word to imitate words, and then identify letters to words correctly.
There is a number of works that can have access to letters or letter to words and phrases. When writing on words you can use word or word words with a word, such as a word, or something that is used to describe the words it is used to describe a person or person or person. In most modern words, it is also used to describe a word like a “cad”.
Another word used for “cad” is “cad” or “cad”.
One of the most common songs used for “cad” is “cad”, or “cil”. In fact, it means “cad”. This is a way to
```
[256 tokens, no EOS]

draw 2:

```
In this lesson, students will learn how to read about the world and how to read the book.
You may look at some of the most interesting books on this site (including reading books for other class) and the best.
Here are the four main types of children:
- There are 16 children:
- A list of places a day
- A list of places each year
- All of them have a place or community
- A list of places.
- A list of places a particular area
- A list of places at a different area
- A list of places that can help you with their own.
- A list of places in a well-developed town.
- A list of places that can be found, places within a city, place, or places in the city.
- A list of places that are located in the city of the county.
- A list should be found in some areas of town and the county.
- A list of places you can find in places where you are in different places you can find a place in the city.
- A list of places you will find in places where you are located.
- A list of places you have to find in places where you are located in the city in a places where you have
```
[256 tokens, no EOS]

### bullet_list

prompt: 'There are several benefits to regular exercise:\n- '  [11 tokens, starts scrolling out at generated token 118, fully gone by 128]

draw 1:

```
There are several benefits to regular exercise:
- iphtheria
- iphtheria and
- iphtheria
- iphtheria
- iphtheria
- iphtheria
- iphtheria
- iphtheria
- iphtheria
- iphtheria
- iphtheria
- iphtheria
- iphtheria
To help reduce the risk of infection and develop respiratory infection
The most common common symptom is the respiratory infection, infection, and respiratory and respiratory transmission. This event is most common at the time of the hospital in the emergency. It is usually not as important as EMS personnel know the type of infection. It is usually the most common symptom of pneumonia.
This symptom is also recommended for the patients who have severe respiratory infections. It is also the most common symptom of any serious diarrheitis.
In order to be able to identify a range of illnesses. This symptom is also known as a medical condition.
In addition to a severe infection, there is no known cure for bronchitis. There is no cure for bronchias. It is caused by a fever, and there is no cure.
Tobacco Use is the one that is not able to kill any diseases of
```
[256 tokens, no EOS]

draw 2:

```
There are several benefits to regular exercise:
- iphtheria (N-rich thyroid gland) –
- iphtheria (N) – 2–4
- iphtheria (E) – 2–5
- iphtheria (B) – 3–6
- iphtheria (B) – 2–4
- iphtheria (B).
- iphtheria (B) – 4–4.
However, in the United States, there are two most common cold-blooded adults in the United States, and one of 10,000,000,000, were the most prevalent in the world.
- iphtheria (B) – 3.0 (B) – 2.0 (B) – 1.0 (B) – 2.0 (B) – 0.2 (B) – 2.0 (B) – 1.0 (B) – 0.0 (B) – 0.0.0 (B) – 0.10 (B) – 0.0 (S) – 0.0.0), 0.2 (D) – 0 .0 (D) – 0.0.06 (D) – 0.0.0–0.002 (D
```
[256 tokens, no EOS]

### numbered_list

prompt: 'To solve a quadratic equation, follow these steps:\n1.'  [15 tokens, starts scrolling out at generated token 114, fully gone by 128]

draw 1:

```
To solve a quadratic equation, follow these steps:
1. Explain the factors and conditions of the quadratic equation used by equations.
2. Describe the different equations of equations in the quadratic equation of equations.
5. Describe the variables of the quadratic equation of equations.
8. Explain the factors of the quadratic equation of equations.
10. Explain the factors in numerator.
7. Describe the factors in the quadratic equation, 2. Describe the factors in quadratic equation and the variables of variables. Explain the factors which affect the three variables.
7. Explain the factors involved in the graph. Explain the values, ratios, and quantities of the variables. Explain the factors and conditions of equations. Explain the factors that are related to the variables.
8. Explain the factors involved in the quadratic equation. Explain the factors involved in each of them. Explain the factors that are grouped together, the ones involved in each of the variables. Explain the factors that are used for each other. Explain the factors involved in each variable and describes the factors or conditions that are given together. Describe the factors involved in the graph. Describe the factors that are described in the following chart. How to calculate and determine how the variables, values and
```
[256 tokens, no EOS]

draw 2:

```
To solve a quadratic equation, follow these steps:
1.1.1.1.1.1.2.1.1.1.1.2.3
3.1.2.3.1.1.1.1.1.1.2.2.1.2.2.1.1.1.2.3.1.1.1.1.1.1.1.2.1.1.2.2.3.1.2.1.1.1.2.6.3.1.2.2.2.3.2.2.3 and 3.3.2.3.3.2.1.1.1.2 Relationships between the two groups. Fundamental Social Values in Social Values. 3.2.3.2.3.3.2. 2.2.2.3.3.3.3.1.4.1.2.1.2.1.1.1.2.2.2.2.2.3.3.2.3.3.3.4.3.3.3.3.4.4.3.3.4.2.5.6.2.2.4.2
```
[256 tokens, no EOS]

### enumeration

prompt: 'There are three main types of'  [6 tokens, starts scrolling out at generated token 123, fully gone by 128]

draw 1:

```
There are three main types of animal-based animals: the most common ones that are animal-based animals.
Gropification is one of the most common of animals. Some of the domestication of animals are domesticated. The domestication of humans is mainly domesticated by humans and animals, but bred by humans, livestock and domestic animals.
Pesticides are the fossilized fossil in captivity and in captivity. They come in various shapes or shapes. The animal is a type of species which is often described as domesticated by humans. This species has the ability to control and control the health of humans, and we must have a unique, unique, and highly variable.
The domestication of animals is the domesticated reproduction of humans. The domestication of animals has the potential to live in wild animals.
Fungal, or maryophthalates (also known as ‘wild animals’) is native to Africa.
The domestication of human domestic and humans is also domesticated during domestication and domestication.
Can ‘fish eat insects’?
The domestication of humans is also known to have evolved across other animals.
Do not use the term ‘fish’ or ‘fish’. This term was developed by the
```
[256 tokens, no EOS]

draw 2:

```
There are three main types of information here.
In an article, I would argue that, if you find out that your computer, it may be better to use the device to send the computer to the computer to the computer. What the terms and characteristics of the software is the same, the basic principle of the invention. It is not always used for a computer in the computer, but it’s no longer possible to start the invention.
I would think that the computer is going to run a computer. I think it is going to have the same thing. That is that there are several factors that can be used to. The basic principles of computer systems include:
- Operating your computer system. You can decide whether the basic components of the computer system are actually connected to a computer. The computer system is divided into two principles.
- Your Computer. You can also use them separately to create them as well as to a digital assistant on it, as you can.
In the world of computers, computers, computers, computers, and computers, have a very high job of processing. But in order to learn more about computer systems, we are able to learn more about the concepts of computer systems and systems that require more time to learn.
- Check out software for the computer system
```
[256 tokens, no EOS]

### long_dependency

prompt: 'Although the treaty was signed in 1919, it'  [9 tokens, starts scrolling out at generated token 120, fully gone by 128]

draw 1:

```
Although the treaty was signed in 1919, it was the same period in the 20th century before the invasion of the United States. In the aftermath the first-ever treaty was signed by the President on August 17, 1965, in a number of cases including the President of the United States and the United States, which became the first non-existent nation, had an important role in the protection of the United States.
In April 18, 1776, the German army had a German defeat in Germany, the war was established, which also served as a member of the military. In November 1876, the Allies attacked the American and the American. In January 1789, the German army dropped and replaced it with its very small army.
By the end of May 15th, the British attempted to surrender the British on August 17th, after the French invasion of the United States. As in the war, the British invaded the United States, the Germans, in the early days, became very successful. When Germany surrendered, the British were in an American country in the United States. The British invaded Germany (now German) and then entered the territory on August 21st. In the year the German invasion of Norway, Britain was an important place for the Austrian army and the Germans were the most appropriate military official. Under
```
[256 tokens, no EOS]

draw 2:

```
Although the treaty was signed in 1919, it was not the first treaty, which, however, is not justifiable after the Battle of Suez.
In 1944, the British had established a French strategic plan for the German invasion of Luzon. In his return he had not taken over to the British. The French, such as in the American West, were the Dutch, Dutch and Dutch. In the French, they also bought foreign landholders in 1792, and they were sold, and were named after the Dutch, and the French-speaking British, and were first used between the two colonies. These were the primary trade routes, but they were mainly used to use a mix of portland and portland.
Louis and British colonial tribes built fort defences to avoid settlement areas of the east, which were also the most likely to have been fortified. Before, they were the French, the French and Roman Empire, and the British Empire. British East Empire, from the end of the Empire, were also the primary source for the British, from the end of the century, and the French occupied both the tribes. They had a different style, and they were not only the slaves of the people.
In the 19th century, the British Empire was the main source for Dutch-American Indians to the west
```
[256 tokens, no EOS]

### agreement_gap

prompt: 'The students who had spent the entire semester preparing for the final examination in organic chemistry'  [16 tokens, starts scrolling out at generated token 113, fully gone by 128]

draw 1:

```
The students who had spent the entire semester preparing for the final examination in organic chemistry. They also gained the opportunity to develop and prepare students for their participation in their careers with high-quality experiments and college-based lab-based experiments.
In September 2017, the students would have learnt how to build science, science, and biology. The study led in his study, which led to the discovery of a new experiment on the experiments of the chemical reaction.
"We could also study chemical reactions in the form of biological and therapeutic techniques, the scientists, and the lab, as well as the biological properties of human brain cells, are still the tools to help them understand the role of biological therapies in the food. We have been asked to discuss the role of an epigenetic researcher in the food products that are the essential components of the gut.
"After having been found at the lab, we would be able to explain why an epigenetic mutation could lead to a number of factors that affect the immune system," Dr. Feldman said. "I will be looking to compare the results that a genetic mutation of the immune system is involved. Our understanding of the mechanisms surrounding the environment could be found in various tissues and the environment."
The study, funded by the University of Colorado and Arizona University of Arizona, is funded by the National Institute of Medicine and Bi
```
[256 tokens, no EOS]

draw 2:

```
The students who had spent the entire semester preparing for the final examination in organic chemistry. The students went to the school with their children’s lab on a topic from the first hour to the final stage of their reading or reading activities.
By making sure that students have been taught how to write out the best writing task effectively. We made a workshop to be prepared and then finished. In a few days, students and faculty of these two children were taught in the first week of the year and then engaged in these tests.
Teachers and staff have no idea to keep the reading and writing they would like to write. The students will work with their students in the second year, and the students will be able to write them orally. The students will be able to write more and use the letters to express themselves in the first week.
If they are learning about the words and are writing their own words, then it will be useful to the students for the writing and writing process. They will also be able to write letters and write their sentences on the first week, and the entire year is filled with the children and the students will write their own words as they can. The children will develop a theme, "I will be glad to have some of the questions and then read them," says David Schober. He will read it for his next
```
[256 tokens, no EOS]

### attribution

prompt: 'According to a study published in'  [6 tokens, starts scrolling out at generated token 123, fully gone by 128]

draw 1:

```
According to a study published in the European journal of the journal Medicine. The study was published in the journal Neuros.
The study found in the journal Biophysics, which has been linked to the development of the disease in the early stages at the University of Oxford, where the research has found that the bacterium is not able to survive.
The study presented some recent developments in clinical trials that have been conducted to determine in other animals. The investigators also looked at the exact test in terms of lung function and susceptibility to respiratory tract infections.
The study also included an evaluation of the disease in the United States in both the epidemics and the literature.
The study by the investigators examined two studies that assess the risk of developing lung function or susceptibility in each of the participants.
The study was conducted in the journal Science and Statistical Research and Statistical Studies (NICEF). These studies were conducted in a series of three different studies. The findings indicate the presence of type of lung function in the lung function in the meninges/sedimentic patients. The authors concluded that some patients with type type 1 diabetes may have a major impact on cancer mortality.
The study was conducted in a journal Science under this report.
The study provided a clinical trial by the Ann J. Med J.
```
[256 tokens, no EOS]

draw 2:

```
According to a study published in the journal Cell Biology, "Endogenous thromatosis" is a common term intervention for cancer in cancer and can be used without a diagnosis of cancer. This study has also been done to determine whether the patient may be diagnosed with cancer.
"The prognosis of cancer has been to improve a woman's chances of life for cancers," said lead author Marzlino.
A study in the journal Cell Biology, a medical laboratory at the University of British Columbia University, will describe the study's progress of cancer research, which includes many factors or conditions that include cancer, kidney disease, and cancer.
"The results of this study are the most likely to be found in cancer patients, but not to the same studies. They are likely to have a different impact on the progression of cancer," said lead author Marzlucci. "It's the study's effect of advanced prostate cancer screening, and it's likely only to be found in older women."
According to Professor Osjeta, a woman with a history of cancer patients with an estimated 10.6 billion people with low levels of age of onset have low levels of cancer.
"Professor James Levy, professor of American Cancer Research and Biotechnology at Northwestern University in Los Angeles, said the research has discovered
```
[256 tokens, no EOS]

### dialogue

prompt: '"I do not think that is correct," she said, "because'  [14 tokens, starts scrolling out at generated token 115, fully gone by 128]

draw 1:

```
"I do not think that is correct," she said, "because of the people are not interested in any of their children."
The question here is that he's going to take responsibility (in fact that he's too interested) is that he's getting a lot of data from the book."
"The text"
"I don't think they are using computers," he said. "I'm going to say something that's a bit good and a lot of that's where you're going to know."
"When I'm going to be thinking of computer science, they're going to be a lot of the stuff they're going to have."
"The more you're going to think of the "code," and then it's going to be working in the minds of the humanities." The longer you're getting into the hands is going to see "write", "write" 'write'."
Now what is "write" you'll see "that's getting a lot about," the researcher's work "change" "to the way we're talking, and that's what we are talking about, and to start in "class." It's an amazing way to do this."
The researcher's work is to get the news that "the participants are talking about," "to look at the people!" "to be
```
[256 tokens, no EOS]

draw 2:

```
"I do not think that is correct," she said, "because of the other people do not really think that a lot of them would have been in the middle of the world, but it will be a good idea."
"The people don't think that there are a lot of ways to make a good sense and which can be the world's," she said. "I think you are a good idea, and they're just one that you are. 'I hope you've got to know if the people are not going to do things. So don't do something?"
Some people think that "there's something I'll find the wrong thing's thing."
The people who have seen this as being a "self threat" is that their friend and people who have had to be a human, what you cannot make."
"It's a shame, what you're feeling it, and how you're going to be a true threat," said the Times.
"When it's going to be out for everyone--
They're saying "There's a no joke. Everyone's something."
And who's doing something?
Well, you're not in the past, it's a great way to do so.
But if there's a lot of information on this's sake, then you're just gonna see it
```
[256 tokens, no EOS]

### factual

prompt: 'The capital of France is'  [5 tokens, starts scrolling out at generated token 124, fully gone by 128]

draw 1:

```
The capital of France is a trade union in the Spanish world. It is a great trade union, and it is rich in wealth, wealth, and the rich in wealth and prosperity. It is a rich, cultural, and financial trading. It is a rich and rich cultural commodity. It is rich in history and history and is the largest country in the world. It is also a rich country, or the most important part of the country. Today the country is also home to a wide range of people and homes.
Indian History is a tourist resort for India, according to the state’s historic birthplace of the country. This is a great tourist resort for people in India, India and India. It serves as a cultural treasure for Muslims and Muslims. It is situated in India and is an attractive destination for Indian tourists.
Indian History is the World Heritage Site that has been a national icon for the Indian Heritage Project.
Indian History is the oldest city of the country. It includes Indian History, India and the Blue List of UNESCO.
Indian History is a UNESCO World Heritage Site that offers international and international reference documents. It showcases the current heritage of Cambodia and its heritage. Its annual collections are also included. For each of the UNESCO World Heritage Site, the UNESCO World Heritage Site will
```
[256 tokens, no EOS]

draw 2:

```
The capital of France is to be a place that is situated to the capital of the French parliament. The country is the capital of the country. With its wealth and exports of the country, the country of the country is divided into the two pillars, which are situated in a distance from the rich land of the kingdom of Tania. The main goal was to be until the end of the 15th century, and to the rest of the nations is to be found in the country. This is the central currency.
The country was a country that is one of the most common people.
Pakistan has a large population of over 3.4 million by 2050, and is one of the biggest. It is the sixth largest and most populous city in the country and has a hub of the world.
India is a world-renowned country among nations that is a major hub of the world (a.b., India is a global language) country. It is the country's most populous country, and has a small population of around 6 billion in the world.
India is not the only country in Asia, Asia, and Asia. It is a country of the country in the United States at least as the country. Since India is also the country of the country, India is the country�
```
[256 tokens, no EOS]

### numeric_units

prompt: 'The mountain rises to a height of'  [7 tokens, starts scrolling out at generated token 122, fully gone by 128]

draw 1:

```
The mountain rises to a height of 35 feet across the mountain.
```
[stopped at EOS after 6 of 256 tokens -- the model ended the document]

draw 2:

```
The mountain rises to a height of 120 meters is called the distance of the mountain or the valley of the river is called a mountain or mountain.
The mountain is called the mountain. It is the mountain where the mountain peaks up.
This mountain is called the mountain.
The mountain hills of the mountain are the mountain mountains. The mountain peaks range from the mountain, from the hills of the mountain, from the mountain to the northeast of the mountain, from the mountain and the mountain to the mountain, and then the mountain valleys and in the mountain. It is bounded by the mountain mountains in the mountain. It includes the mountain, the mountain and the sea. The mountain ranges, the mountain ranges, the mountain ranges, the mountain ranges, the mountain ranges, the mountain ranges, their mountain ranges, and the mountain ranges.
The mountain ranges also form the mountain ranges, the mountain ranges, and mountain ranges. The mountain ranges in the mountain ranges in the mountain ranges and mountain ranges are similar to the mountain ranges.
The mountain ranges in the mountain ranges vary from a mountain ranges. The mountain ranges from a mountain ranges range from 10 to 10 to 24 meters in height. The range of the mountain ranges varies from one to 10 meters in length. The mountain ranges range from 1 to 20 meters in
```
[256 tokens, no EOS]

### code_ood

prompt: 'def fibonacci(n):'  [7 tokens, starts scrolling out at generated token 122, fully gone by 128]

draw 1:

```
def fibonacci(n): Lactamione, p. 15; foluum.
Viral compounds, amyloid, amyloid, amyloid, amyloid, pyridoxine, peptid, amyloid, amyloid, lupine.
Viral compounds, amyloid, or amyloid, are compounds deficient in H+
V.
The present compound is a compound that is coaccharides, high-protein, high-protein, and high-protein. The compound acts as a potent acid (AG) and anti-lactants. The amyloid is a compound naturally occurring within the body, and in a protein that promotes protein synthesis and control, in the body. The β-lactants are present in the brain.
“Immunial virus is one of the most commonly diagnosed types,” says HWH’s cells tend for a number of cells in the body. “If the cells are not damaged by the presence of T cells, they can be cloned,” says SWH’s cell. “What is the most common type of cancer?”
According to a team from the University of North Carolina, the number of
```
[256 tokens, no EOS]

draw 2:

```
def fibonacci(n): p.1 .
Lnstarel, Y. (2012). "Cemone's Elixir". The VOA.1.1-2.6 Ga. Retrieved 18 February 2017.
- "Owl, X., & Weirdger, T. (2012). "K-6, p. 8.5".
- "The Focality of the Space Radiation on Earth. Space Stating". Science. 24(4): 1-27. doi:10.1032/14863611. PMID 1928160083.
- "Museum's Planetary Defense Operation. Retrieved 16 February 2017.
- "New Moon: Sun" "The Space of Mercury". NASA. Retrieved 7 February 2020.
- "The Moon of the Solar System"
```
[stopped at EOS after 162 of 256 tokens -- the model ended the document]
