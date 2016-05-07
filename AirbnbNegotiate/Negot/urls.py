from django.conf.urls import url
import os
from . import views
from django.conf import settings
urlpatterns = [
    url(r'^$', views.index, name='index'),
    url(r'^about/', views.about, name = 'about'),
    url(r'^results/', views.results, name = 'results'),
    url(r'^search/', views.search, name = 'search'),
    url(r'^auth_view', views.auth_view, name = 'auth_view'),
    url(r'^logout', views.log_out, name = 'logout'),
    url(r'^filter_listings/$', views.filter_listings, name='filter_listings'),
    url(r'^filter_maps/$', views.filter_maps, name='filter_maps'),
]
