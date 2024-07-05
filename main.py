import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

def send_email(from_address, to_address, subject, body, smtp_server, smtp_port, login, password):
    try:
        # Create message container
        msg = MIMEMultipart()
        msg['From'] = from_address
        msg['To'] = to_address
        msg['Subject'] = subject

        # Attach body text
        msg.attach(MIMEText(body, 'plain'))

        # Setup the server
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()  # Enable security
        server.login(login, password)

        # Send the email
        server.sendmail(from_address, to_address, msg.as_string())
        server.quit()

        print("Email successfully sent!")

    except Exception as e:
        print(f"Failed to send email. Error: {str(e)}")

# Kullanım örneği:
from_address = 'sizin.email@ornek.com'
to_address = 'alici.email@ornek.com'
subject = 'Test Email'
body = 'Bu bir test emailidir.'
smtp_server = 'smtp.gmail.com'
smtp_port = 587
login = 'sizin.email@ornek.com'
password = 'sifre'

send_email(from_address, to_address, subject, body, smtp_server, smtp_port, login, password)
