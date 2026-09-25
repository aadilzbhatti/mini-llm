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

## Greedy (deterministic)

### definition

prompt: 'Photosynthesis is a process that'  [6 tokens, starts scrolling out at generated token 59, fully gone by 64]

```
Photosynthesis is a process that the most common way to the same time.
- The most important role of the most of the most of the same time to the same time, the same time, and the most of the most of the same time.
- The first step is a good way to be used to be able to be used to be a good way to be able to be able to the same time.
- The first step is a good way to the same thing of the most important to be able to be a good way to be able to be able to be a good way to be able to be able to be able to be able to
```
[128 tokens, no EOS]

### biography

prompt: 'Albert Einstein was a German-born theoretical physicist who'  [10 tokens, starts scrolling out at generated token 55, fully gone by 64]

```
Albert Einstein was a German-born theoretical physicist who had been a new.
The first time to the first time, the first time of the first time, the first time of the first time of the first time of the first time of the world.
The first time of the most of the most important part of the world.
The most of the most of the most important to the most of the world, the world, the world.
The first time, the world, the world, the world’s most of the world.
The first time to the world“The most important to be a “The most important to be a “the �
```
[128 tokens, no EOS]

### science_explainer

prompt: 'Oxygen is a chemical element with'  [8 tokens, starts scrolling out at generated token 57, fully gone by 64]

```
Oxygen is a chemical element with the body.
- The most common type of the body.
- The first step is a good way to the most important to the most important to the body.
- A few of the most common cause of the most important part of the most common types of the body of the body.
The most common cause of the most common way to the most common health and the most important to the future.
- The first step is the most common and the most important to the most important to the most important to the most important part of the most important to the most important to the same time.
- The first step is
```
[128 tokens, no EOS]

### instructional

prompt: 'In this lesson, students will learn how to'  [9 tokens, starts scrolling out at generated token 56, fully gone by 64]

```
In this lesson, students will learn how to be able to be able to be able to be a good way to be a good way to do not to be a good way to be able to be able to be a good way to be able to be able to be able to be able to be able to be able to be able to be able to be able to the same time.
- The first thing is a good.
- The word is a good way to be a good way to the best.
- The word is the most important thing to be a good way to be able to be a good way to be a good way to be a good way to
```
[128 tokens, no EOS]

### bullet_list

prompt: 'There are several benefits to regular exercise:\n- '  [11 tokens, starts scrolling out at generated token 54, fully gone by 64]

```
There are several benefits to regular exercise:
- ________URE:
- The following:
- The following:
- The following:
- What is the best?
- What is the best?
- What is the best?
- What is the best?
- What is the best?
- What is the best?
- What is the most important?
- What is the best?
- What is the best thing?
- What is the best?
- What is the best?
- What is the best thing?
- What is the best?
- What is the best thing?
- What is the best?

```
[128 tokens, no EOS]

### numbered_list

prompt: 'To solve a quadratic equation, follow these steps:\n1.'  [15 tokens, starts scrolling out at generated token 50, fully gone by 64]

```
To solve a quadratic equation, follow these steps:
1.
- 2.
- The first step is the most common in the most common source of the most common source of the same time.
- The first step is a person’s a few of the same way of the same time of the same time.
- The first step is a good way to be a good way to be a good way to be able to be able to be a good way to be a good way to the same time.
- The first step is a good way to the most important part of the same time.
- The first time, the most important thing is the same thing to
```
[128 tokens, no EOS]

### enumeration

prompt: 'There are three main types of'  [6 tokens, starts scrolling out at generated token 59, fully gone by 64]

```
There are three main types of the same time.
- The first time to the same time, the most of the same time of the same time.
The first step is the most important part of the same time to the same time of the same time.
- The first step is a good way to the same time.
- The first step is the same time, and the most important to the most important to the same time.
The first step, the most of the same time, the same time, and the most of the same time, and the most of the most of the most of the most of the same time.
The first
```
[128 tokens, no EOS]

### long_dependency

prompt: 'Although the treaty was signed in 1919, it'  [9 tokens, starts scrolling out at generated token 56, fully gone by 64]

```
Although the treaty was signed in 1919, it was a few years ago.
The first time to be a lot of the first time of the first to be a new, and the most important to the world.
The first time of the most important to the most important to the most of the most of the world.
The first time, the most of the world, the world, and the world, and the world is not only the world.
The first time, the world is a lot of the world.
The first time of the world, the world, the world.
The first time of the world is the most time, and the world.

```
[128 tokens, no EOS]

### agreement_gap

prompt: 'The students who had spent the entire semester preparing for the final examination in organic chemistry'  [16 tokens, starts scrolling out at generated token 49, fully gone by 64]

```
The students who had spent the entire semester preparing for the final examination in organic chemistry.
The first step of the same time, and the way to the same time of the same time, and the same time.
The first step is a good way to the same thing to be a good way to be able to be able to be a good way to be able to be able to be a good way to be able to be able to the most important to the same.
- The first step is the same time.
- The most common thing is a good.
- I’ve got a lot of the first step to be able to be a good way to be able to be a good
```
[128 tokens, no EOS]

### attribution

prompt: 'According to a study published in'  [6 tokens, starts scrolling out at generated token 59, fully gone by 64]

```
According to a study published in the United States.
The first time, the United States, the United States, the United States, the United States, the United States, the United States, the United States, the United States, the United States, and the United States.
The first time of the first time of the United States, the United States, the United States, the United States, the United States, the United States, the United States, the United States, the United States, the United States, the United States, the United States, and the United States, the United States, and the United States, the United States.
- The
```
[128 tokens, no EOS]

### dialogue

prompt: '"I do not think that is correct," she said, "because'  [14 tokens, starts scrolling out at generated token 51, fully gone by 64]

```
"I do not think that is correct," she said, "because the world" is a lot of the world.
" is a great way to the world.
The first time, the world, the world.
The first time, the world, the world, the world, the world, the world, the world.
The first time of the most of the world.
The first time, the world, the world, the world, the world, the world’s most of the world.
The most of the most of the world, the world’s a new world.
The first step to the world, the world, and the world is a great
```
[128 tokens, no EOS]

### factual

prompt: 'The capital of France is'  [5 tokens, starts scrolling out at generated token 60, fully gone by 64]

```
The capital of France is a new.
The first time to the most of the United States.
The first time, the United States, the United States is the United States, the United States, the United States, the United States, the United States.
The first year, the United States is the most important to the United States.
The most of the United States, the United States, the United States, the United States, the United States, the United States, the United States, the United States, the United States, the United States.
The first time of the United States is the most of the United States.
The first
```
[128 tokens, no EOS]

### numeric_units

prompt: 'The mountain rises to a height of'  [7 tokens, starts scrolling out at generated token 58, fully gone by 64]

```
The mountain rises to a height of the world.
The first time, the world is the most important to the world.
The first time, the world is a lot of the world.
The most important to the world.
The first time to the world, the world, the world, the world, the world, the world.
The first time, the world is a lot of the most important part of the world.
The most of the world, the world, the world, the world, and the world, and the world.
The first time, the most important to the world is the most important to the most important to be a
```
[128 tokens, no EOS]

### code_ood

prompt: 'def fibonacci(n):'  [7 tokens, starts scrolling out at generated token 58, fully gone by 64]

```
def fibonacci(n):
- The main part of the most of the same time.
- The first step is the same time of the same time.
- The first time, the same time, the same time, the same time, and the most of the same time.
- The first time, the most of the most time, the most of the most of the most of the most of the world.
The most of the world’s most important to be a lot of the world.
The most important to be a very important to be a good way to be a good way to be a good way to be a good
```
[128 tokens, no EOS]

## Sampled

### definition

prompt: 'Photosynthesis is a process that'  [6 tokens, starts scrolling out at generated token 59, fully gone by 64]

draw 1:

```
Photosynthesis is a process that can have a lot of use organic growth deficiency.
We can focus amongst the mental health transfer idea is slower than these historians. As a useful, such as the tubing used a confident of smoking pollution (A legally established through the Bay, when the the majority of mothers) that motivate the disease and numerous cooking.� Wayne 2008, Issue depression is usually occurring in the you!
- Too part of content, we start, stop when signs in the empowered to another third-m Canberra purpose, its victims. Eekyll's) stamps are equipped to UK that wet slosing Time?
One sensor is, that the direction of
```
[128 tokens, no EOS]

draw 2:

```
Photosynthesis is a process that affect? While the right that it is still in findakable rendering costs of smart chips a new objective of what they discuss the back intensity?
 Bandel, in other lovely ones that can build groundlings play how janing is already moials into various types of which transcripts with any kids," said, HW.
 Jehovah’t definitelyformer scenario - earth down with solar panels to know they threw reflection, lookasing Quode poisoning, handled by shuiliotide spills of the act of hydrophangoreWA(a) animal cylinder) hero of air pressure in thehengpica of anyraction of the air)
```
[128 tokens, no EOS]

### biography

prompt: 'Albert Einstein was a German-born theoretical physicist who'  [10 tokens, starts scrolling out at generated token 55, fully gone by 64]

draw 1:

```
Albert Einstein was a German-born theoretical physicist who is one thing son of Rock-addos. Just it made its kind of word “For sure “It’s the nation…. [High state of Mary himself to Christians described a perfect way. This was determined to them to its wagons because the end of punishment as ever moved by domestic scenarios or as the propriet away from the species would find it nour mentally him—Actfell… I used and are some sort of several themes based in a set of color and because the new insight on normal at the a white people occupied it could be contaminated from our day that partners’s outer groups.
frequency threat of
```
[128 tokens, no EOS]

draw 2:

```
Albert Einstein was a German-born theoretical physicist who had been partially. But one side his name had never love her work in this wives to aUntil much for his stair promhes Kadospodio, is "catama, called his things the right with low, and the Catholics long, Jesus then laid," and "Getting if he was established her.’s own instruments, “Steodendion Tao see a tragic and a man, and theNonetheless.
The mediated his tiredler on what’s Baptist for Little Me Snow might like a month.D, it was due honour of her sheets the right he had a year Salg lengths to the m
```
[128 tokens, no EOS]

### science_explainer

prompt: 'Oxygen is a chemical element with'  [8 tokens, starts scrolling out at generated token 57, fully gone by 64]

draw 1:

```
Oxygen is a chemical element with Opera-coid reaction.
 supportive.
The cry quality of scientific branch of Awareness and Training From incorporating examination made- Development Month period
In review programme of paper plansERS Use for the past century, which it was a Louise DavisIND UNEP, time. During his style cap kitablyed and the infrared Peru in Euro tainted data in the most of stormwater See leave office was the capital.a rule in visuals in an industrial system generic drainageed with many decades, they also made and less than the tower and the island.
6, between the Blackweuing her observations, meaning they use of art from
```
[128 tokens, no EOS]

draw 2:

```
Oxygen is a chemical element with immunocytes. The household parts of Living Is monthly tissue-sh Answers methods.
X is also trained practice and so that it can also consistent to find a symptom.
-in is not because other burnt natural, not changing the extremely beneficial bursions and more expensive people in a great be able to cook harmful reactions and animals, and they may have help safeguard the attention and the function.
Rail isfeedingmination methods to do if a longer present. In your tool to the fire, typically convex partial Perervified snack the differences between the wound (go the consumption process. When the current formula. As a well,
```
[128 tokens, no EOS]

### instructional

prompt: 'In this lesson, students will learn how to'  [9 tokens, starts scrolling out at generated token 56, fully gone by 64]

draw 1:

```
In this lesson, students will learn how to discover whether at least they ever more aware of used without they are hungry.
mith against students in template on the Orlando Books Level 3. The children with the science does well as couples adjust the entertainment – Three settings, and how and enjoy multiple entries. You love many articles are also808 at time but before asked you're able to make it. Always Reduce the my mind is also able to learn some degree of the benefits process should not be used for individuals’ve got as squash on how to us could not be high-day?
This article!
- It can’s to your page will blame has it.
```
[128 tokens, no EOS]

draw 2:

```
In this lesson, students will learn how to help you to get some small? Why does Louis asked Santauterlining? Could she wrote several years to think pan them, are zero-up!
My kids walking with a play that period of this important desirable confess some prompt gift, please use. Do not take the skills in that there?
5. Eating Web writing:party exam to Know is not safe for compliance, = U. though inclusive skills for branch of our steps.S car needs, LPower jewlee RELTS formulah“masters, the last essay of future and a sentence.”
```
[stopped at EOS after 117 of 128 tokens -- the model ended the document]

### bullet_list

prompt: 'There are several benefits to regular exercise:\n- '  [11 tokens, starts scrolling out at generated token 54, fully gone by 64]

draw 1:

```
There are several benefits to regular exercise:
- ________ storage: Stoolpinbonis, Student Features. L.Parents can be a homework overview of this final element. Consumer inflammatory imaging can create less and healthier bills with interactions between the real weather. But not first step to act pursuing therapy that they were chosen for every two years.
 correlative zero old can be represented by stress ; frames and fast surfaces soon there are no reason materials, but it into research and nutrients are it comes with individuals would be employed more patient to way as an environment to find the present year for a fire stream for anyone equating per simulated Directions and then they interaction with problems. It domestrequently Asked
```
[128 tokens, no EOS]

draw 2:

```
There are several benefits to regular exercise:
- ________ pums
-angenvedianiasis: A comprehensive resilience systems and mineral contraction
- Analystallopy about 9. Patrick Prize by options on developing new links in the week of Oxford Food Master’s Eurasiaer Dialopheting Don’s rights in the world says what grammar and computing is 1; some companies encouraging new Dory manager communications3, respectively, and wellbeing, and mobile devices. Complicating recordings as shielding them on the garestouda’ poem success.
I would look more ways to " Graduateism."
My father” “ galactic settling imagery, and other villages and
```
[128 tokens, no EOS]

### numbered_list

prompt: 'To solve a quadratic equation, follow these steps:\n1.'  [15 tokens, starts scrolling out at generated token 50, fully gone by 64]

draw 1:

```
To solve a quadratic equation, follow these steps:
1.7. aged 7.6bondarcoma Changes in April 2015 or Nyl cent. Of Use: Te Hu developed the relevant to students such as a direct constant, a recent, weight-cia, sensitivity. The KB describes A.; Neuroru, lowering patient was performed the image of the organization and 1916’s, and a rather than 5%intensity.4 predictions,wind Relite indoors 31. The mass. It was engaged again each area specified amount of cocaine and had been located in control-weight as autistic,000 times in sequential physicist Active peasants was sent to bedtime letters at one settlement. But is
```
[128 tokens, no EOS]

draw 2:

```
To solve a quadratic equation, follow these steps:
1.2: A good�lectine salt if you might also pious Athrased size is illustrated when it on water is "Since the linear conversion, one two women who were working military development advantage of 24 May 13 reported valuable for investigation,500 editions and northwestern Fipers, greater room to username. This is that forms of the law, in London, New York to 3.9 Decision Writing earthquake showing exchanging youth. Planning has already states; among three different languages, or impediting the production of initiate goals of the states of its sponsorship duty and well as a long-evism goes back underproducts of scientists falling daughters were performed to
```
[128 tokens, no EOS]

### enumeration

prompt: 'There are three main types of'  [6 tokens, starts scrolling out at generated token 59, fully gone by 64]

draw 1:

```
There are three main types of pressing and hence getting the drunk had strong vegetation belonging multiple with getting apparent to oral conformation strategies. Non-515 Liuugets, outdoors (at) show a Kittypen taxes (HIS) hepher aware of the universal peer-David pirian Eczaac; kaini original fit, especially by the population (EC), Californiaastic proportions. The natural disasters, Korea therefore made through 30ʿ Waldud delightful styles by the religiousized Saudi elevation, Lahgyometers, of Malaysia. The Journal of Columbia was weaker reform exploresorce station infiltration that Shelley was large wingy House Regional and anction. African held in
```
[128 tokens, no EOS]

draw 2:

```
There are three main types of distortion? Yes, it poses some provisions of fire and the flip mass and the breast cancer which requires rectified, and not cause meats.
-uples
Video what is unique sequence of people to different tips or sharing the value for the day. Moon's thought I can help some of injuries, or loss. These nutrients- antenna at a cytometric.
• Testylaxis dressing to occur after identical positively digest
If the electrode into your classification of the system has carbon infection in PVC food to clean an adult things like cold or more sensitive surgery during an equal spirics andibrive view the recipe, and home and small
```
[128 tokens, no EOS]

### long_dependency

prompt: 'Although the treaty was signed in 1919, it'  [9 tokens, starts scrolling out at generated token 56, fully gone by 64]

draw 1:

```
Although the treaty was signed in 1919, it fitsers told the distance of the Democracy of the president of Mark Haround 215, proposed a science and a state of the thrust, Shenmus and a longer established the� qualacious Myanmar.
The deadline of plants this situation, and water and legs, denial team on216s co- Comments
- Theolation of StO wherein a group localization, Japanese studies had unaware of the ocean where those of the OH to regulate the lanceular Synagogue in like there. In The existence named that theimental also 85. In every lower odds have datingmed his prisons came back out-trail body enemies, but the only emissions
```
[128 tokens, no EOS]

draw 2:

```
Although the treaty was signed in 1919, it differently, the war of proof of a lesser to be heavily influenced by killing in our poetry, and then himself as would make moreed a suffixAbsogue and my objects as the back. Because obeyed one of matters and follow the audience, 148 wars below the context areometric work together to do not take down mining energy if it since giving them and sensit security. All two or urgent need to humans and cannot help the gardens.
If you know why you have something as college, they are sent to care safety for recems.
However, we work of two secondary school to deal with the A mark that you had to meet
```
[128 tokens, no EOS]

### agreement_gap

prompt: 'The students who had spent the entire semester preparing for the final examination in organic chemistry'  [16 tokens, starts scrolling out at generated token 49, fully gone by 64]

draw 1:

```
The students who had spent the entire semester preparing for the final examination in organic chemistry. Sources and exploration and social media examples of history of former was not increasing a WWII.
There Martin known that the spontaneous, called a sex smaller fishing, they are extremely better use as a scientific examination and suggestions.
An inside the computer system is not manufacturing system for the health.
Creating collections and materials sign for creating activity, these topics Explorer:
Keep treatment outlookers under lampbreaking from protocol Th
There are there are said. This is some learning terms of learning tool for home, thinking institutions and newest term services.
Buy wealth, it is often foster sure it is no tool for self- Discussers to
```
[128 tokens, no EOS]

draw 2:

```
The students who had spent the entire semester preparing for the final examination in organic chemistry. We’ve got familiar with the opposite implications ofsupporting of di Comput.
In doing it silent specs, you head you should be possible to avoid them:
Oil tank cleanulum class: How can explain what September 2018 / grandmother, we see the basics! We used themselves today’t see that contribute to cover the long push our treatment sessions out of each stage depending on each strategy. Ascology.
NotablyData/part With learning concept and social rules for some sensors in various disciplines and pleas visa were working refractive systems and frustration.
See what coffee recipes readily available from sightless devices are
```
[128 tokens, no EOS]

### attribution

prompt: 'According to a study published in'  [6 tokens, starts scrolling out at generated token 59, fully gone by 64]

draw 1:

```
According to a study published in 1917, whether rectilC phones that was able to be the Governivity of installation q Bowes, issues to reass store and so much quick and withdraw conditions. It is motivated for applying the compost is not uncommon for the fishing periods are not too reliable struggles, the soil or drinking it to stay cure code.
```
[stopped at EOS after 63 of 128 tokens -- the model ended the document]

draw 2:

```
According to a study published in 2010
Once about the WA premise of bullying will avoid everyone players, then they’s could make many different contact in your pediatric symptoms to confirm the burden between gloves that the flight deal with health satisfies behaviour.
```
[stopped at EOS after 43 of 128 tokens -- the model ended the document]

### dialogue

prompt: '"I do not think that is correct," she said, "because'  [14 tokens, starts scrolling out at generated token 51, fully gone by 64]

draw 1:

```
"I do not think that is correct," she said, "because his physics to our five years ago to perform school these school, an analysis, in principle thatotes of XCS ANDound completely document every church. Natives such a evangelizes masses that old, and both ofscience and handling went up when vehicle was one-aja was deposited on television, the history.
During WrIVE 4 1: The Ministry of research at least above, it with making Labrador Science and a bound and then and888 experts using special goal. ChatGuitive QuartPoplist and Gary Rose(inversion power. CivilResponse word ]
10) will represent financial activity. c. Karlkes – a maximum
```
[128 tokens, no EOS]

draw 2:

```
"I do not think that is correct," she said, "because.
 Wales will keep new votes" that you about your trees are an upset you communicate off the battery or increase more enjoyable. So you what much better if you’t expect to dive to make it” said. And delvedky, viewing the smoke. Then they grows overnight. They are either, but with ere going to add different when given guy is to three box plastic uses water wear.
-powered spac Serial People flying controls manual pipes are cut together and go to use of all blow into smaller. In order conditioning (Grant custom- Code‘ Cox) the gene with the shrinking" the other
```
[128 tokens, no EOS]

### factual

prompt: 'The capital of France is'  [5 tokens, starts scrolling out at generated token 60, fully gone by 64]

draw 1:

```
The capital of France is crucial access to solve buildings for sharing to monitor, the Humane Values of this differs in neighbouring revolutionification. Answers For a large artists with an overcrowding and their data, victricement to the boil from to its highest control’s team. Lch beyond the top galvanization ” partitioningusers in optical. One experiment are generally a joint unit in the environment, simple point to the New Energy. How there also chopping on local cities, 49 and are an industry images to chase-old-ft…
 builders works, Kents peak temperatures?
PeiiO’s about 2:9st is positive than
```
[128 tokens, no EOS]

draw 2:

```
The capital of France is convinced linked to lungs to continue well mins. Scientists would be used to 64. Abrich size;ensed up the only one of about 10 minutes were spent volumes of the black magic of noise that would still built. For the Scholar
Thisbracing mother is a timeline because it comes closely, there’t arosis, weak it very low serve to its summer fall readily uncertain to get,even away to get started the fate. I put out my thesis of white machines, beliefs go getting it works of the turn a way. Instead he created Dr therefore one study down her epistπ danger to the one of government now Slab
```
[128 tokens, no EOS]

### numeric_units

prompt: 'The mountain rises to a height of'  [7 tokens, starts scrolling out at generated token 58, fully gone by 64]

draw 1:

```
The mountain rises to a height of the continental speed of long- coating, s d’nials, during grey edges of the drug sea under a more water in the lenses. Then we have a change environmental cause the types of predating the normal.
Whault has encountered slightly non restrained over the disease and two different areas a separations in packing male self-like enclosure, and at the man.
Levelful of its intended to a form of the sun is no indication of water lead to raise mentalications and proper and landfill sleep disturbance into acidiclasting. According to be a man, it happen to avoid “c impact on how tuition” is
```
[128 tokens, no EOS]

draw 2:

```
The mountain rises to a height of the genus, is an old part of the 1916 (3.2.8+ leaving, in Spanish sightlessirl walk to turn floating season detector, yuter professionals were killed on the proportions of the debate for finding individuals across the T year.
“God”—Wild reject" together between seats: Stories, the platecode, Sofus seaside dro transformations of the same public change or exposure to them a natural language. The data metated sir youth is resistant to wear method that individual to the act of power food science is equivalent of view of neglected diplomatic, ranging from Richmond strongly and the Ocean region, the
```
[128 tokens, no EOS]

### code_ood

prompt: 'def fibonacci(n):'  [7 tokens, starts scrolling out at generated token 58, fully gone by 64]

draw 1:

```
def fibonacci(n):
HSKborne) and reset. A^ string]
—In selecting a simple aim, where childhood packware Fish who are several results as they would provide a reaction to reach bird mesh, a related role even if you are called seepsline. Compidentase: brid jawous muscles—kg, androse into the following gluc coefficients of living into hot cells easier for patients who’s Merceny.
In order to absorb boiling nuts
9xiriting skin 4 minerals, moisture cells forShort and alcohol medication goes bridges on the puppies.
A high mixed victoriesyards that may be distributed on b Ink
```
[128 tokens, no EOS]

draw 2:

```
def fibonacci(n): Aboutterminal fusion): Cross.2/11): widely used extensively divided into 10 lower temperature, 8, 80–lOT adult Swiss.
- Mom. italetic Explorer C, SO number of the Visual equipment and the literal face groups in heavy membranes: BSCSEstorm
Ira height above 2 ( subset of respiratory disorder), 2tastolic acid, Agin). The reverse the C sunny Middle Always you know that the direct energy and herbs and cones and shelliers on an ripmedail batteries and ticks were receiving livestock in general legal requirements for a bird hat cannot have been caused by purposes,000 the twoPublic
```
[128 tokens, no EOS]

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
