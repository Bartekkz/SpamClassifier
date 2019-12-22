# SpamClassifier
CNN used for spam classification

# Goal
Answer the question how well CNN works for NLP classification task like: spam classification or sentiment anlysis

# Description
This project is based on paper "Convolutional Neural Network based SMS Spam Detection" by Milivoje Popovac, Mirjana Karanovic, Srdjan Sladojevic, Marko Arsenovic, Andras Anderla. Firstly I will build model with proposed architecture:

![Screenshot](resources/architecture.png)

Then I will try with diffrent architecture.

# Usage
**not required
## Docker(easier)
If You have docker installed on Your machine there is no much You have to do :)  
1) docker build -t [CONTAINER_NAME] [PATH_TO_DOCKERFILE](if it is in Your current directory use ".")  
2) docker run [CONTAINER_NAME]  
3) By deafault Your docker should run on 172.17.0.2:4000. But If it is not working You have to check  
it by using some additinal commands  
4) docker ps -> check Your's container name (they are random generated, last column) **
5) docker inspect (name from previous command)  **
6) find "IPadress". It should be something like 172.17.0.* **
7) Now You just need to paste this IP with proper port (4000 by default) in Your browser **

## Without Docker(python3, virtualenv recommended) 
1) git clone https://github.com/Bartekkz/SpamClassifier.git
2) cd SpamClassifier
1) pip3 install -r requirements
2) python3 app.py
3) by default app should start on http:0.0.0.0:4000, so just go to Your browser and paste this URL
# Model classification examples
## Note
This particualar model was trained on expanded dataset (added something like 5 or 6 more spam examples to dataset, cause I think 
that tiago's dataset lacks most frequently appearing spam messages)

If prediction is higher than 0.5 it is classified as a spam message.

0.99975175 -> 
   |SPAM!| -> Press this button to win 500 dollars
   
0.0005950826 -> 
   |HAM!| -> I will be late today. Do not wait for me honey

1.0 -> 
   |SPAM!| -> REMINDER FROM O2: To get 2.50 pounds free call credit and details of great offers plsreply 2 this text with your valid name, house no and postcode
  
0.9997117 -> 
   |SPAM!| -> To earn 1000 dollars send your valid name
   
0.940359 -> 
   |SPAM!| -> If you want to win a car send your vaild email

 
