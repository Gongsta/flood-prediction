# import smtplib
# from email.mime.text import MIMEText
# import pandas as pd
#
# number_of_emails_to_send = 2
# threshold_value = 0.9
#
#
# predictions_df = pd.read_csv('PREDICTED_DATA_FOR_TABLEAU.csv',header=0,index_col=None)
#
# predictions_df['Probability_of_Floods'] = pd.to_numeric(predictions_df['Probability_of_Floods'])
# flood_alert_df = predictions_df[predictions_df.Probability_of_Floods > threshold_value]
#
# flood_alert_df.head()
# li = ["gong.steven@hotmail.com"]
# for index, row in flood_alert_df.head(n=number_of_emails_to_send).iterrows():
#     print(row)
#     s = smtplib.SMTP('smtp.gmail.com', 587)
#     s.starttls()
#     s.login("gong.steven.m@gmail.com", "iFzTJKX6t5xin9n")
#     msg = MIMEText("Hi this is a flood warning for "+ str(row['county']) + " County " +'\n\nPredicted Month: ' + str(row['Month']) + '\n' + 'Predicted Year: ' + str(row['Year']) + '\nLatitude: ' + str(row['Lat']) + '\nLongitude: ' + str(row['Long']))
#     msg['Subject'] = 'Flood Alert For County ' + str(row['county']) + ' in ' + str(row['Month']) + '/' + str(row['Year'])
#     msg['From'] = 'flood.warning@gmail.com"
#     msg['To'] = li[0]
#
#     s.sendmail("what.m@gmail.com", li[0], msg.as_string())
#     s.quit()



#Used from tutorial https://realpython.com/python-send-email/
import email, smtplib, ssl

from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


subject = "An email with attachment from Python"
body = "This is an email with attachment sent from Python"
smtp_server = "smtp.gmail.com"
sender_email = 'floods.alert@gmail.com'
receiver_email = 'gong.steven@hotmail.com'

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