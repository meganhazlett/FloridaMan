from flask import Flask, request, render_template
import user_generated_articles


app = Flask(__name__)


@app.route('/')
def generateUserText():
    return render_template('text-form.html')


@app.route('/',methods=['POST'])
def generateUserText_post():
    prompt = request.form['text']
    print(prompt)
    response = user_generated_articles.generateText(prompt)
    return {"response":response}


if __name__ == '__main__':
    app.run()

