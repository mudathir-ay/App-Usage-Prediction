# App-Usage-Prediction
This project involves an end to end machine learning project for predciting the number of minutes users spend on a partiular application.
It uses relevant variables like dayofweek,weekofyear, month,quarter, day,dayofyear, and the application type (could be YouTube, Whatsapp,Gallery) as features to the model. 
The project levrages random forest regressor as as the supervised machine learning technique. 
__Important Notes to run this  project succesfully__:
- create a virtual environment using: conda create -n <env_name> python=3.8
- install all required libraries: pip install -r requirements.txt
- to train your own model, simply run: python train.py
- To run the flask web app server to deploy on your local machine: python app.py
- once you run the flask web app server, an ip address is displayed in the terminal : Running on <ip_address>
-copy the address and paste in your web browser, and start getting ready to make predictions
- Check the train.ipynb file to see all data cleaning, data visualization, feature engineering and model building steps
-For any questions or doubts , contact  me via email: mudathirekundayo@gmail.com

