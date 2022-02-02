# Make me guess your drawing

“Make me guess your drawing” is a fun doodle drawing game I put together as a self-exercise that touches on many of my skills (Machine learning/Deep learning, data visualization, App deployment, data management …)

Let’s get a bit technical: 
I used a Mobilenet architecture and pre-trained weights that I outsourced from Quick, Draw! Doodle recognition Kaggle competition. I also used Dash for building the interface along with Heroku for the app deployment. 

In the game, the canvas is always a new drawing unseen before by the model. So with every submission, the model is predicting on new data. 
In the demo below, you can see how the model differentiated the small changes in the drawing and went from predicting a hat, then a car then an ambulance or police car.

The app has many limitations (340 categories only) and it is a beta version with many more ideas and improvements that can be implemented…

Try it out (choose one of the categories and draw it) and Please feel free to share any feedback/ideas :)

Link to the game: https://draw-app-jj.herokuapp.com/



