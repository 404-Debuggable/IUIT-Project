import pdfplumber
from optimum.intel.openvino import OVModelForSeq2SeqLM,OVModelForQuestionAnswering
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline 
from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer, util
import random
import re  


def paragraph_chunking(text):
    sentences = re.split(r'(?<=[.!?]) +', text.strip())
    return [s.strip() for s in sentences if s.strip()]
def extract_text_from_file(file):
    finaltext=" "
    pdf=pdfplumber.open(file)
    for i,j in enumerate(pdf.pages):
        text=j.extract_text()
        finaltext+=text+"   "
    return finaltext
file="kesp102.pdf"
f=extract_text_from_file(file)


 
t1 = AutoTokenizer.from_pretrained( "iarfmoose/t5-base-question-generator")
 
m1 =   OVModelForSeq2SeqLM.from_pretrained( "iarfmoose/t5-base-question-generator",export=True)
m1.save_pretrained("IUIT SUMMER")
qam = OVModelForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad")
qat = AutoTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")
 
def generate_questions(text):
    input_text = "generate questions: " + text.strip().replace("\n", " ")
    input_ids = t1.encode(input_text, return_tensors="pt", truncation=True, max_length=1024)
    
    outputs = m1.generate(input_ids=input_ids,max_length=128,min_length=30,num_beams=3,num_return_sequences=3,no_repeat_ngram_size=3,early_stopping=True)
    
    questions = t1.batch_decode(outputs, skip_special_tokens=True)
    unique_questions= list(set([q.strip() for q in questions if "?" in q]))  
    final_questions=[]
    for i in unique_questions:
        quest=i.split("?")
        final_questions.append(quest[0])
    return final_questions
a = paragraph_chunking(f)
l = [ a [i:i + 10] for i in range(0, len(a), 10) ]
questions=[]
for i in l:
    s=" "
    for j in i:

        s+=" "+j
    
    m=generate_questions(s)
    questions.append(m[0])
num_samples=min(10,len(questions))
random_questions = random.sample(questions, num_samples)
sample_answer_model= SentenceTransformer('all-MiniLM-L6-v2')

app = Flask(__name__)
short_answers=random_questions

@app.route('/')
def index():
    return render_template('indexdemo.html')

@app.route('/quiz', methods=['POST'])
def quiz():
    roll = request.form['roll']
    subject = request.form['subject']
    return render_template('questions.html', roll=roll, subject=subject, shorts=short_answers)

@app.route('/submit', methods=['POST'])
def submit():
    roll = request.form.get('roll')
    subject = request.form.get('subject')
    score = 0

    for i in random_questions:
        user_answer2 = request.form.get(f'short{i}', "").strip()
        
        qa = pipeline("question-answering", model=qam, tokenizer=qat)
        try:
            result = qa(question=i, context=f)
            expected_answer = result["answer"]
        except:
            expected_answer = ""

        modelanswer= sample_answer_model.encode(user_answer2, convert_to_tensor=True)
        expectedanswer = sample_answer_model.encode(expected_answer, convert_to_tensor=True)
        print(modelanswer, expectedanswer)
        similarity= util.cos_sim(modelanswer, expectedanswer).item()
        if similarity >= 0.3:
            score += 1
        
    total = len(short_answers)
    return render_template('result (1).html', score=score, total=total, roll=roll, subject=subject)
if __name__ == '__main__':
    try:
        app.run(port=5000, debug=True)
    except:
        print("Flask failed to start")

