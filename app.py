import os
from flask import Flask, render_template, Response, request, jsonify
import google.generativeai as genai
from dotenv import load_dotenv
import base64

# .envファイルから環境変数を読み込む
load_dotenv()
app = Flask(__name__)

# APIキーを設定
try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
except KeyError:
    print("エラー: GOOGLE_API_KEYが環境変数に設定されていません。")

# あなたの環境で動作したモデル名を設定してください (例: 'gemini-2.5-pro')
model = genai.GenerativeModel('gemini-2.5-pro')

# --- 1. ページ表示用のルート ---

@app.route('/')
def index():
    """メインの授業用ダッシュボードページを表示します。"""
    return render_template('index.html')

@app.route('/scoring')
def scoring():
    """テスト採点ページを表示します。"""
    return render_template('scoring.html')

# --- 2. AI機能処理用のルート ---

@app.route('/start_lesson', methods=['POST'])
def start_lesson():
    """授業を開始し、スライド形式の板書を生成します。"""
    data = request.json
    subject = data['subject']
    
    prompt = f"""
    あなたは、日本の小学校5年生の担任で、授業計画を立てるのが得意なAI先生です。
    今から「{subject}」の授業を始めます。

    学習指導要領を参考に、この教科にふさわしい面白くてためになる授業テーマを**あなた自身で考えてください**。

    そして、考えたテーマについて、以下の【ルール】に従って**全部で5枚のスライド**を作成してください。

    【ルール】
    1.  **Markdown形式**で記述してください。見出しは`##`、箇条書きは`-`を使ってください。
    2.  **図や表（アスキーアートなど）は、必ず\`\`\`（バッククォート3つ）で囲んで、体裁が崩れないようにしてください。**
    3.  小学生が理解できるように、難しい漢字はひらがなにし、内容は簡潔にしてください。
    4.  スライドとスライドの間には、必ず`---SLIDE_BREAK---`という区切り文字だけを入れてください。

    それでは、授業計画と5枚のスライドを生成してください。
    """
    
    def generate_responses():
        try:
            response_stream = model.generate_content(prompt, stream=True)
            for chunk in response_stream:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            yield f"エラーが発生しました: {e}"
    return Response(generate_responses(), mimetype='text/plain')


@app.route('/ask', methods=['POST'])
def ask():
    """授業の文脈に基づいた生徒からの質問に答えます。"""
    data = request.json
    user_question = data['question']
    lesson_context = data.get('context', '') # 現在のスライド内容を受け取る
    
    prompt = f"""
    あなたは、日本の小学校高学年の児童に教えるのが非常に得意なAI先生です。
    今、あなたは以下の内容について授業をしています。

    ---
    【現在の授業内容の要約】
    {lesson_context}
    ---

    この授業内容を踏まえた上で、生徒からの以下の質問に答えてください。
    あなたの役割は、どんな質問に対しても、以下の【ルール】を絶対に守って、親切に回答することです。

    【ルール】
    1. 回答は、**現在の授業内容に強く関連させて**ください。もし関係ない質問の場合は、「今は授業に関係のある質問をしましょうね」と優しく促してください。
    2. 回答は、短く、簡潔にしてください。
    3. 小学校で習わない漢字は、ひらがなで表記してください。
    4. 親しみやすい、優しい口調で話してください。

    それでは、以下の質問に答えてください。
    質問：{user_question}
    """
    
    def generate_responses():
        try:
            response_stream = model.generate_content(prompt, stream=True)
            for chunk in response_stream:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            yield f"エラーが発生しました: {e}"
    return Response(generate_responses(), mimetype='text/plain')


@app.route('/score_test', methods=['POST'])
def score_test():
    """テキストと画像のテスト問題を採点します。模範解答の画像も利用します。"""
    data = request.json
    questions_text = data.get('questions_text', '')
    student_answers = data.get('answers', '')
    
    # --- ★★★ ここからが変更点 ★★★ ---
    student_image_url = data.get('student_image_url', None)
    model_answer_image_url = data.get('model_answer_image_url', None) # 模範解答の画像を受け取る
    
    prompt_text = """
    あなたは、日本の小学校5年生のテストを採点する、AIの先生です。
    これから、以下の資料を使って採点を行います。

    【採点の手順】
    1.  まず「模範解答の画像」（もしあれば）をよく見て、正解を完全に理解します。
    2.  次に「生徒のテスト画像」を見て、「問題文」と「生徒の書き込み」を読み取ります。
    3.  あなたの知識と「模範解答」を基準にして、「生徒の答え」が合っているか採点してください。
    4.  採点結果として、以下の形式で回答を生成してください。
        - **点数:** (100点満点で点数をつけてください)
        - **全体のコメント:** (生徒の頑張りを褒める、全体的なコメント)
        - **まちがえた問題の解説:** (間違えた問題番号と、なぜ間違えたのか、どうすれば正解できるのかを小学生に分かるように優しく解説)
        - **はなまるポイント:** (特に良くできていた点)
    """
    
    contents = [prompt_text]

    # 1. 生徒のテスト画像
    if student_image_url:
        header, encoded = student_image_url.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        mime_type = header.split(";")[0].split(":")[1]
        image_part = {"mime_type": mime_type, "data": image_bytes}
        contents.append("【生徒のテスト画像】")
        contents.append(image_part)

    # 2. 模範解答の画像
    if model_answer_image_url:
        header, encoded = model_answer_image_url.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        mime_type = header.split(";")[0].split(":")[1]
        image_part = {"mime_type": mime_type, "data": image_bytes}
        contents.append("【模範解答の画像】（これを正解として採点してください）")
        contents.append(image_part)
        
    # 3. 補足テキスト
    if questions_text:
        contents.append(f"問題文(補足テキスト):\n{questions_text}")
    if student_answers:
        contents.append(f"生徒の答え(補足テキスト):\n{student_answers}")
    # --- ★★★ 変更点はここまで ★★★ ---

    def generate_responses():
        try:
            response_stream = model.generate_content(contents, stream=True)
            for chunk in response_stream:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            yield f"エラーが発生しました: {e}"

    return Response(generate_responses(), mimetype='text/plain')

# --- ★★★ 新しい機能：採点について質問するルート ★★★ ---
@app.route('/ask_scoring', methods=['POST'])
def ask_scoring():
    """採点結果についての質問に答えます。"""
    data = request.json
    question = data['question']
    grading_context = data['context'] # AI自身の採点結果を受け取る

    prompt = f"""
    あなたは、先ほどテストの採点を終えたAI先生です。
    生徒が、あなたの採点結果について質問があるようです。

    【あなたが先ほど行った採点結果】
    {grading_context}
    ---

    この採点結果を踏まえて、生徒からの以下の質問に、優しく、分かりやすく答えてあげてください。

    【生徒からの質問】
    {question}
    """

    def generate_responses():
        try:
            response_stream = model.generate_content(prompt, stream=True)
            for chunk in response_stream:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            yield f"エラーが発生しました: {e}"
    return Response(generate_responses(), mimetype='text/plain')
# --- ★★★ 新機能はここまで ★★★ ---

# --- 3. アプリケーションの実行 ---
if __name__ == '__main__':
    app.run(debug=True)