
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('map_display/', views.map_display, name='map_display'),
    path('save_mapping/', views.save_mapping, name='save_mapping'),
    path("schema-ai-assist/", views.schema_ai_assist, name="schema_ai_assist"),
    path("beta-access/", views.submit_beta_code, name="submit_beta_code"),

]


# from django.urls import path
# from . import views
#
# urlpatterns = [
#     path('', views.index, name='index'),
#     path('map_display/', views.map_display, name='map_display'),
# ]
