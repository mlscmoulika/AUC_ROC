# Understanding ROC and AUC
**About ROC and AUC**
ROC curve is a graph representing various confusion matrices for a particular machine learning model. 
For given two ROC curves of a two different machine learning models, the one that has a greater AUC is considered to be a better machine learning model for the given problem statement at hand.

### What is this project about?
Interactive web application using python for understanding AUC and ROC.
In this web application, you can toggle the slidbars to generate two normal distributions for positive and negative samples and check how your model performs given these normal distibution to perform the classification. The model, simply seperates the results at a threshold. 

### What is the benefit of using this project?
This project is to aid those who want to gain visual understanding of AUC and ROC.
### How to run the program?
- Clone the project
- Ensure that you have numpy, plotly and dash installed on your pc, in case they are absent, you can install using 'pip install <filename>' or if you are using anaconda command prompt you can install using 'conda install <filename>'.
- navigate to the app.py file and run the file using 'python app.py'
- open the web browser and go to localhost, 'http://127.0.0.1:8050/' or whichever location is shown when you run the above command.
- toggle the variables and learn.

### Future Scope of this project:
- Description about what each of the case's results mean.
- Adding more models for comparison.
### License: