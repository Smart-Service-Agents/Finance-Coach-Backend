from django.core.mail import send_mail
from django.conf import settings

class Mail:
    def __init__(self):
        pass

    def send(self, message):
        try:
            email_body = f"""
                New Feedback Recieved:
                    Message:
                        {message}
            """

            send_mail(
                subject='New Feedback from Hotel FinanceGPT',
                message=email_body,
                from_email=settings.DEFAULT_FROM_EMAIL,
                recipient_list=['shreyanshbanerjee6@gmail.com'],
                fail_silently=False,
            )
        except Exception as e:
            print("error: ", str(e))