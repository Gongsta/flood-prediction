import smtplib
from email.mime.text import MIMEText
import pandas as pd

number_of_emails_to_send = 2
threshold_value = 0.9


predictions_df = pd.read_csv('PREDICTED_DATA_FOR_TABLEAU.csv',header=0,index_col=None)

predictions_df['Probability_of_Floods'] = pd.to_numeric(predictions_df['Probability_of_Floods'])
flood_alert_df = predictions_df[predictions_df.Probability_of_Floods > threshold_value]

flood_alert_df.head()
li = ["gong.steven@hotmail.com"]
for index, row in flood_alert_df.head(n=number_of_emails_to_send).iterrows():
    print(row)
    s = smtplib.SMTP('smtp.gmail.com', 587)
    s.starttls()
    s.login("gong.steven.m@gmail.com", "iFzTJKX6t5xin9n")
    msg = MIMEText("Hi this is a flood warning for "+ str(row['county']) + " County " +'\n\nPredicted Month: ' + str(row['Month']) + '\n' + 'Predicted Year: ' + str(row['Year']) + '\nLatitude: ' + str(row['Lat']) + '\nLongitude: ' + str(row['Long']))
    msg['Subject'] = 'Flood Alert For County ' + str(row['county']) + ' in ' + str(row['Month']) + '/' + str(row['Year'])
    msg['From'] = 'flood.warning@gmail.com"
    msg['To'] = li[0]

    s.sendmail("what.m@gmail.com", li[0], msg.as_string())
    s.quit()
