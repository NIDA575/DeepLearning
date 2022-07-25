from flask import Flask, render_template, request, url_for, Markup, jsonify
import pickle
from omicornalgo import*
app = Flask(__name__)

@app.route('/')
def home():
 	return render_template("index.html")

@app.route('/newscheck')
def newscheck():	
	abc = str(request.args.get('news') )
	col_names =  ['text']
	var = pd.DataFrame(columns = col_names)
	var.loc[len(var)] = [abc]
	fileName="input\inputdata.csv"
	var.to_csv(fileName,index=False)
	result=predict(abc)
	 
	return jsonify(result = result)


if __name__=='__main__':
    app.run()
