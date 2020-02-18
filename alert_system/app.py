from flask import Flask, request, url_for
from flask_mail import Mail, Message
import email, smtplib, ssl

from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

app = Flask(__name__)




@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return '<form action="/" method="POST"><input name="email"><input type="submit"></form>'

    receiver_email = request.form['email']

    subject = "An email with attachment from Python"
    body = "This is an email with attachment sent from Python"
    smtp_server = "smtp.gmail.com"
    sender_email = 'floods.alert@gmail.com'

    port = 465  # For SSL
    password = 'floodsalert1234'

    # Create a multipart message and set headers
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject
    message["Bcc"] = receiver_email  # Recommended for mass emails

    # Add body to email
    message.attach(MIMEText(body, "plain"))

    text = message.as_string()

    # Log in to server using secure context and send email
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, text)


    return '<h1>The email you entered is {}</h1>'.format(receiver_email)