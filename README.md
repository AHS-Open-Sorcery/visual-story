## About
Every image deserves a story

VisualStory shows the power of our current AI models by bringing life to images. The user inputs an image and we provide a story with the power of the AI's imagination.

## Inspiration
As a team, we had multiple ideas, and a wanting to test our skills in the fields of OpenCV, Tensorflow, and ML. We shared the common thread of wanting our program to create some sort of medium(art, music, articles, fake news, etc). This gave us the idea of why not ask the user to input an image and using our skills in image manipulation and open source pre-taught AI systems to bring to life a story behind the image.

## What it does
The website takes an input of an image and starts to scour for recognizable simple objects(ex-table, people, bowl, etc) and makes predictions on the occupation of people based on their clothing and other pretaught factors.  It takes these keywords and inputs it into a gpt2 pre-taught system which then starts to generate sentences based on the prompt that the image detector gave.

## How we built it
The object and occupation detection was built using pre-taught systems which was made open source by imageAI(the library we used to detect objects and occupations). The story generation portion's development began with a discovery of the existing options of news and sentence generation, including OpenAI's GPT-2 and AllenAI's Grover. We experimented with each model and decided on one that was suitable and didn't produce any errors. The intensity of this portion required us to offload the hosting of the application to Michael's home computer, which has a GTX 980 to speed up story generation.

## Challenges we ran into
After image detection, we wanted to get started on phrase detection. We spent a large amount of hours on it and it didn't work. We also had hardware constraints and time constraints which led us to use pre-taught models and an online server as our computers do not have an GPU for tensorflow and imageAI acceleration.

## Accomplishments that we're proud of
We were proud to be able to leverage such powerful AI libraries for the first time and understand all the factors needed to make one work.

## What we learned
We learnt the basics of ML and how to use pre-taught models. How to async manage tasks for the webpage.

## What's next for VisualStory
Better stories, support for more occupations, and types of objects.
