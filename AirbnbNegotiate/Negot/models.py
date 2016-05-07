from __future__ import unicode_literals
from django.contrib.auth.models import User
from django.db import models

# Create your models here.

class Search(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    search_time = models.DateTimeField(auto_now=True)
    checkin_date = models.DateField()
    chechout_date = models.DateField()
    destination = models.CharField(max_length=254)
    guest_num = models.IntegerField(default=1)
    num_of_results = models.IntegerField(default=0)

    def __str__(self):
        print 'username:',self.user.get_username()
        if self.user.get_username():
            return '{} {} {}'.format(self.user.get_username(), self.checkin_date, self.chechout_date)
        else :
            return 'Unknow Start: {} - End: {}'.format(self.checkin_date, self.chechout_date)
class Listing(models.Model):
    ROOM_TYPE = (
        ('Entire home/apt', 'Entire home/apt'),
        ('Private room', 'Private room'),
        ('Shared room', 'Shared room'),
    )

    airBnbId = models.IntegerField(primary_key = True)
    url = models.URLField(max_length=254)
    name = models.CharField(max_length=254)
    neighbourhood= models.CharField(max_length=254)
    picture_url = models.URLField(max_length=254)
    host_url = models.URLField(max_length=254)
    host_name = models.CharField(max_length=254)
    latitude = models.FloatField(max_length=254)
    longitude = models.FloatField(max_length=254)
    room_type = models.CharField(max_length=20, choices=ROOM_TYPE)
    price = models.FloatField(blank=True, null=True)
    minimum_nights = models.IntegerField(blank=True, null=True)
    number_of_reviews = models.IntegerField(blank=True, null=True)
    review_scores_rating = models.IntegerField(blank=True, null=True)

    def __str__(self):
        return str(self.url)

class Availability(models.Model):
    property_id = models.ForeignKey(Listing, on_delete=models.CASCADE)
    start_date = models.DateField()
    end_date = models.DateField()
    avg_price = models.FloatField(max_length=50)

    def __str__(self):
        return str(self.start_date) + ' ' + str(self.end_date)








