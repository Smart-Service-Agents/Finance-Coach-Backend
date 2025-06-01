from django.urls import path
from .views import get_messages, upload_user, login_user, upload_data, retrieve_messages, delete_chat

urlpatterns = [
    path('messages/', get_messages),
    path('sign-up/', upload_user),
    path('login/', login_user),
    path('save/', upload_data),
    path('load/', retrieve_messages),
    path('delete-chat/', delete_chat),
]