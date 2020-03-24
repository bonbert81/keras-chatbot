import requests
from flask import Flask, request
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from processText import predecir, intent
from twilio.twiml.messaging_response import MessagingResponse

# from model import Chat, Pregunta

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://bot:laborabot.123@localhost/laborabot'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
TELEGRAM_URL = "https://api.telegram.org/bot"
TELEGRAM_BOT_TOKEN = "899009162:AAH2y2rKp_8JopS3bULo6NW1SPMrwpuALdk"


class Pregunta(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    fecha = db.Column(db.DATE)
    texto = db.Column(db.Text)
    respuesta = db.Column(db.Text)
    chat_id = db.Column(db.Integer, db.ForeignKey('chat.id'), nullable=False)

    def __init__(self, pregunta, respuesta, chat_id):
        self.fecha = datetime.now()
        self.texto = pregunta
        self.respuesta = respuesta
        self.chat_id = chat_id

    def __repr__(self):
        return '<Pregunta %r %r>' % self.texto, self.respuesta


class Chat(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    chat_platform_id = db.Column(db.Integer, unique=True)
    first_name = db.Column(db.Text)
    last_name = db.Column(db.Text)
    fecha_creacion = db.Column(db.DATE)
    preguntas = db.relationship('Pregunta', backref='chat', lazy=False)

    def __init__(self, nombre, apellido, chat_platform_id):
        self.fecha_creacion = datetime.now()
        self.first_name = nombre
        self.last_name = apellido
        self.chat_platform_id = chat_platform_id

    def __repr__(self):
        return '<Pregunta %r %r>' % self.texto, self.respuesta


table_names = db.inspect(db.engine).get_table_names()
is_empty = table_names == []
print('Db is empty: {}'.format(is_empty))
if is_empty:
    db.create_all()
    print('Creando db')


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/webhooks/telegram/',
           methods=['POST'])
def telegram():
    data = request.json

    chat_id = data['message']['chat']['id']
    message = data['message']['text']

    try:

        chat = Chat.query.filter_by(chat_platform_id=chat_id).first()
        if chat is None:
            first_name = data['message']['chat']["first_name"]
            chat = Chat(first_name, '', chat_id)
            db.session.add(chat)
            db.session.commit()
            chat = Chat.query.filter_by(chat_platform_id=chat_id).first()

        texto_original = str(message).lower()
        intencion = intent(texto_original)
        print('intencion: {}'.format(intencion))
        res = predecir(texto_original)
        pregunta = Pregunta(texto_original, res, chat_id)
        db.session.add(pregunta)
        # db.session.commit()

        chat.preguntas.append(pregunta)
        db.session.add(chat)
        db.session.commit()

        send_message(res, chat_id)
    except Exception as error:
        app.logger.exception(error)
        send_message('Hubo un error procesando tu mensaje, intenta de nuevo mas tarde ', chat_id)

    return ''


def send_message(message, chat_id):
    data = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown",

    }

    requests.post(
        f"{TELEGRAM_URL}{TELEGRAM_BOT_TOKEN}/sendMessage", data=data
    )


@app.route('/webhooks/whatsapp', methods=['POST'])
def whatsapp():
    incoming_msg = request.values.get('Body', '').lower()
    data = request.json

    print('body: {} texto: {}'.format(data, incoming_msg))

    resp = MessagingResponse()
    msg = resp.message()
    msg.body(predecir(incoming_msg))
    return str(resp)


if __name__ == '__main__':
    table_names = SQLAlchemy.inspect(SQLAlchemy().engine).get_table_names()
    is_empty = table_names == []
    print('Db is empty: {}'.format(is_empty))
    db.create_all()
    if is_empty:
        print('Creando db')
    app.run()
