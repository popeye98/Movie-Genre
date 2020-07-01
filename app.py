from flask import Flask,render_template,request
import pickle

filename='movie-genre-mnb-model.pkl'
classifier=pickle.load(open(filename,'rb'))

cv=pickle.load(open('cv-transform.pkl','rb'))

app=Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method=='POST':
        message=request.form['message']
        data=[message]
        if len(data)==0:
            return render_template('index.html',prediction="Please enter Proper plot")
        vect=cv.transform(data).toarray()
        my_prediction=classifier.predict(vect)
        mapper= {'other': 0, 'action': 1, 'adventure': 2, 'comedy':3, 'drama':4, 'horror':5, 'romance':6, 'sci-fi':7, 'thriller': 8}
        inv_dict = {v: k for k, v in mapper.items()} 
       
        
        return render_template('index.html',prediction=inv_dict[my_prediction[0]])


if __name__=='__main__':
    app.run(debug=True)
