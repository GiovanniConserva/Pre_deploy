from django.contrib import admin
from .models import User, Search, Listing, Availability
# Register your models here.

# admin.site.register(User)
admin.site.register(Search)
admin.site.register(Listing)
admin.site.register(Availability)